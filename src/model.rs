use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{Linear, Module, VarBuilder};
use std::path::Path;

const HIDDEN_SIZE: usize = 768;
const VOCAB_SIZE: usize = 256_302;
const NUM_LAYERS: usize = 12;
const NUM_HEADS: usize = 12;
const HEAD_DIM: usize = HIDDEN_SIZE / NUM_HEADS;
const FF_HIDDEN_SIZE: usize = 2_048;
const TAG_HIDDEN_SIZE: usize = 3_072;
const ANTECEDENT_HIDDEN_SIZE: usize = 3_072;
const MENTION_EMBEDDING_SIZE: usize = HIDDEN_SIZE * 2;
const RELATIVE_POSITION_BUCKETS: usize = 32;
const RELATIVE_POSITION_MAX_DISTANCE: usize = 128;
const RMS_NORM_EPS: f64 = 1e-6;

pub(crate) struct CorpipeRuntime {
    encoder: CorpipeEncoder,
    heads: CorpipeHeads,
}

impl CorpipeRuntime {
    pub(crate) fn load(model_dir: &Path, tag_count: usize) -> Result<Self> {
        let weights_path = model_dir.join("model.safetensors");
        let device = Device::Cpu;

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                std::slice::from_ref(&weights_path),
                DType::F32,
                &device,
            )
        }
        .with_context(|| format!("failed to load {}", weights_path.display()))?;

        Ok(Self {
            encoder: CorpipeEncoder::load(vb.clone())?,
            heads: CorpipeHeads::load(vb, tag_count)?,
        })
    }

    pub(crate) fn encode_input(&self, input_ids: &[u32]) -> Result<Tensor> {
        self.encoder.forward(input_ids)
    }

    pub(crate) fn with_context<F, R>(&self, f: F) -> R
    where
        F: FnOnce() -> R + Send,
        R: Send,
    {
        self.encoder.device().with_context(f)
    }

    pub(crate) fn gather_word_embeddings(
        &self,
        embeddings: &Tensor,
        word_indices: &[usize],
    ) -> Result<Tensor> {
        BatchedEmbeddings::new(embeddings).gather_word_embeddings(word_indices)
    }

    pub(crate) fn tag_logits(&self, word_embeddings: &Tensor) -> Result<Tensor> {
        self.heads.tag_logits(word_embeddings)
    }

    pub(crate) fn antecedent_scores(
        &self,
        embeddings: &Tensor,
        all_mentions: &[(usize, usize)],
        current_mentions: &[(usize, usize)],
    ) -> Result<Tensor> {
        self.heads
            .antecedent_scores(embeddings, all_mentions, current_mentions)
    }
}

struct CorpipeEncoder {
    embed_weight: Tensor,
    layers: Vec<EncoderLayer>,
    final_layer_norm: Tensor,
}

impl CorpipeEncoder {
    fn load(vb: VarBuilder) -> Result<Self> {
        let embed_weight = vb
            .get(
                (VOCAB_SIZE, HIDDEN_SIZE),
                "_encoder.encoder.embed_tokens.weight",
            )
            .with_context(|| "failed to load encoder embedding weight")?;

        let mut layers = Vec::with_capacity(NUM_LAYERS);
        for layer_index in 0..NUM_LAYERS {
            layers.push(EncoderLayer::load(vb.clone(), layer_index)?);
        }

        let final_layer_norm = vb
            .get((HIDDEN_SIZE,), "_encoder.encoder.final_layer_norm.weight")
            .with_context(|| "failed to load final encoder layer norm")?;

        Ok(Self {
            embed_weight,
            layers,
            final_layer_norm,
        })
    }

    fn forward(&self, input_ids: &[u32]) -> Result<Tensor> {
        let mut hidden = self.embedding_lookup(input_ids)?;
        let attention_buckets =
            RelativeAttentionBuckets::for_seq_len(hidden.dims()[1], hidden.device())?;

        for layer in &self.layers {
            hidden = layer.forward(&hidden, &attention_buckets)?;
        }

        TensorOps::rms_norm(&hidden, &self.final_layer_norm, RMS_NORM_EPS)
            .with_context(|| "final encoder rms norm failed")
    }

    fn embedding_lookup(&self, input_ids: &[u32]) -> Result<Tensor> {
        let ids = Tensor::from_vec(
            input_ids.to_vec(),
            (input_ids.len(),),
            self.embed_weight.device(),
        )
        .with_context(|| "failed to create input id tensor")?;

        self.embed_weight
            .index_select(&ids, 0)
            .with_context(|| "embedding index_select failed")?
            .unsqueeze(0)
            .with_context(|| "embedding unsqueeze failed")
    }

    fn device(&self) -> &Device {
        self.embed_weight.device()
    }
}

struct EncoderLayer {
    attention: SelfAttentionBlock,
    feed_forward: GatedFeedForwardBlock,
}

impl EncoderLayer {
    fn load(vb: VarBuilder, layer_index: usize) -> Result<Self> {
        Ok(Self {
            attention: SelfAttentionBlock::load(vb.clone(), layer_index)?,
            feed_forward: GatedFeedForwardBlock::load(vb, layer_index)?,
        })
    }

    fn forward(
        &self,
        input: &Tensor,
        attention_buckets: &RelativeAttentionBuckets,
    ) -> Result<Tensor> {
        let attended = self.attention.forward(input, attention_buckets)?;
        self.feed_forward.forward(&attended)
    }
}

struct SelfAttentionBlock {
    layer_norm: Tensor,
    q_weight: Tensor,
    k_weight: Tensor,
    v_weight: Tensor,
    o_weight: Tensor,
    relative_attention_bias_weight: Tensor,
    layer_index: usize,
}

impl SelfAttentionBlock {
    fn load(vb: VarBuilder, layer_index: usize) -> Result<Self> {
        let prefix = format!("_encoder.encoder.block.{layer_index}.layer.0");

        Ok(Self {
            layer_norm: vb
                .get((HIDDEN_SIZE,), &format!("{prefix}.layer_norm.weight"))
                .with_context(|| {
                    format!("failed to load attention layer norm for layer {layer_index}")
                })?,
            q_weight: vb
                .get(
                    (HIDDEN_SIZE, HIDDEN_SIZE),
                    &format!("{prefix}.SelfAttention.q.weight"),
                )
                .with_context(|| format!("failed to load q weight for layer {layer_index}"))?,
            k_weight: vb
                .get(
                    (HIDDEN_SIZE, HIDDEN_SIZE),
                    &format!("{prefix}.SelfAttention.k.weight"),
                )
                .with_context(|| format!("failed to load k weight for layer {layer_index}"))?,
            v_weight: vb
                .get(
                    (HIDDEN_SIZE, HIDDEN_SIZE),
                    &format!("{prefix}.SelfAttention.v.weight"),
                )
                .with_context(|| format!("failed to load v weight for layer {layer_index}"))?,
            o_weight: vb
                .get(
                    (HIDDEN_SIZE, HIDDEN_SIZE),
                    &format!("{prefix}.SelfAttention.o.weight"),
                )
                .with_context(|| format!("failed to load o weight for layer {layer_index}"))?,
            relative_attention_bias_weight: vb
                .get(
                    (RELATIVE_POSITION_BUCKETS, NUM_HEADS),
                    &format!("{prefix}.SelfAttention.relative_attention_bias.weight"),
                )
                .with_context(|| {
                    format!("failed to load relative attention bias for layer {layer_index}")
                })?,
            layer_index,
        })
    }

    fn forward(
        &self,
        input: &Tensor,
        attention_buckets: &RelativeAttentionBuckets,
    ) -> Result<Tensor> {
        let normed = TensorOps::rms_norm(input, &self.layer_norm, RMS_NORM_EPS)?;
        let q = TensorOps::split_heads(
            &TensorOps::linear_no_bias_forward(&normed, &self.q_weight)?,
            NUM_HEADS,
            HEAD_DIM,
        )?;
        let k = TensorOps::split_heads(
            &TensorOps::linear_no_bias_forward(&normed, &self.k_weight)?,
            NUM_HEADS,
            HEAD_DIM,
        )?;
        let v = TensorOps::split_heads(
            &TensorOps::linear_no_bias_forward(&normed, &self.v_weight)?,
            NUM_HEADS,
            HEAD_DIM,
        )?;

        let scores = q.matmul(&k.transpose(2, 3)?)?;
        let bias = self.relative_attention_bias(attention_buckets)?;
        let probabilities = TensorOps::softmax_last_dim(&scores.broadcast_add(&bias)?)?;
        let context = TensorOps::merge_heads(&probabilities.matmul(&v)?)?;
        let projected = TensorOps::linear_no_bias_forward(&context, &self.o_weight)?;

        (input + projected).with_context(|| {
            format!(
                "attention residual add failed for layer {}",
                self.layer_index
            )
        })
    }

    fn relative_attention_bias(
        &self,
        attention_buckets: &RelativeAttentionBuckets,
    ) -> Result<Tensor> {
        self.relative_attention_bias_weight
            .index_select(&attention_buckets.bucket_ids, 0)
            .with_context(|| "relative bias index_select failed")?
            .reshape((
                attention_buckets.seq_len,
                attention_buckets.seq_len,
                NUM_HEADS,
            ))
            .with_context(|| "relative bias reshape failed")?
            .permute((2, 0, 1))
            .with_context(|| "relative bias permute failed")?
            .unsqueeze(0)
            .with_context(|| "relative bias unsqueeze failed")
    }

    fn relative_position_bucket(relative_position: isize) -> usize {
        let relative_position = -relative_position;
        let num_buckets_half = RELATIVE_POSITION_BUCKETS / 2;

        let mut bucket = 0usize;
        if relative_position > 0 {
            bucket += num_buckets_half;
        }

        let distance = relative_position.unsigned_abs();
        let max_exact = num_buckets_half / 2;

        if distance < max_exact {
            return bucket + distance;
        }

        let distance = distance as f64;
        let max_exact = max_exact as f64;
        let max_distance = RELATIVE_POSITION_MAX_DISTANCE as f64;

        let value = max_exact as usize
            + ((distance / max_exact).ln() / (max_distance / max_exact).ln()
                * ((num_buckets_half as f64) - max_exact)) as usize;

        bucket + value.min(num_buckets_half - 1)
    }
}

struct RelativeAttentionBuckets {
    seq_len: usize,
    bucket_ids: Tensor,
}

impl RelativeAttentionBuckets {
    fn for_seq_len(seq_len: usize, device: &Device) -> Result<Self> {
        let mut bucket_ids = Vec::with_capacity(seq_len * seq_len);

        for query_position in 0..seq_len {
            for key_position in 0..seq_len {
                bucket_ids.push(SelfAttentionBlock::relative_position_bucket(
                    query_position as isize - key_position as isize,
                ) as u32);
            }
        }

        let bucket_ids = Tensor::from_vec(bucket_ids, (seq_len * seq_len,), device)
            .with_context(|| "failed to create relative bucket tensor")?;

        Ok(Self {
            seq_len,
            bucket_ids,
        })
    }
}

struct GatedFeedForwardBlock {
    layer_norm: Tensor,
    wi_0: Tensor,
    wi_1: Tensor,
    wo: Tensor,
    layer_index: usize,
}

impl GatedFeedForwardBlock {
    fn load(vb: VarBuilder, layer_index: usize) -> Result<Self> {
        let prefix = format!("_encoder.encoder.block.{layer_index}.layer.1");

        Ok(Self {
            layer_norm: vb
                .get((HIDDEN_SIZE,), &format!("{prefix}.layer_norm.weight"))
                .with_context(|| format!("failed to load ff layer norm for layer {layer_index}"))?,
            wi_0: vb
                .get(
                    (FF_HIDDEN_SIZE, HIDDEN_SIZE),
                    &format!("{prefix}.DenseReluDense.wi_0.weight"),
                )
                .with_context(|| format!("failed to load wi_0 for layer {layer_index}"))?,
            wi_1: vb
                .get(
                    (FF_HIDDEN_SIZE, HIDDEN_SIZE),
                    &format!("{prefix}.DenseReluDense.wi_1.weight"),
                )
                .with_context(|| format!("failed to load wi_1 for layer {layer_index}"))?,
            wo: vb
                .get(
                    (HIDDEN_SIZE, FF_HIDDEN_SIZE),
                    &format!("{prefix}.DenseReluDense.wo.weight"),
                )
                .with_context(|| format!("failed to load wo for layer {layer_index}"))?,
            layer_index,
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let normed = TensorOps::rms_norm(input, &self.layer_norm, RMS_NORM_EPS)?;
        let gate = TensorOps::gelu(&TensorOps::linear_no_bias_forward(&normed, &self.wi_0)?)?;
        let value = TensorOps::linear_no_bias_forward(&normed, &self.wi_1)?;
        let hidden = gate.mul(&value).with_context(|| {
            format!(
                "ff gate/value multiply failed for layer {}",
                self.layer_index
            )
        })?;
        let ff = TensorOps::linear_no_bias_forward(&hidden, &self.wo)?;

        (input + ff)
            .with_context(|| format!("ff residual add failed for layer {}", self.layer_index))
    }
}

struct CorpipeHeads {
    dense_hidden_tags: Linear,
    dense_tags: Linear,
    dense_hidden_q: Linear,
    dense_hidden_k: Linear,
    dense_q: Linear,
    dense_k: Linear,
}

impl CorpipeHeads {
    fn load(vb: VarBuilder, tag_count: usize) -> Result<Self> {
        Ok(Self {
            dense_hidden_tags: candle_nn::linear(
                HIDDEN_SIZE,
                TAG_HIDDEN_SIZE,
                vb.pp("_dense_hidden_tags"),
            )
            .with_context(|| "failed to load _dense_hidden_tags")?,
            dense_tags: candle_nn::linear(TAG_HIDDEN_SIZE, tag_count, vb.pp("_dense_tags"))
                .with_context(|| "failed to load _dense_tags")?,
            dense_hidden_q: candle_nn::linear(
                MENTION_EMBEDDING_SIZE,
                ANTECEDENT_HIDDEN_SIZE,
                vb.pp("_dense_hidden_q"),
            )
            .with_context(|| "failed to load _dense_hidden_q")?,
            dense_hidden_k: candle_nn::linear(
                MENTION_EMBEDDING_SIZE,
                ANTECEDENT_HIDDEN_SIZE,
                vb.pp("_dense_hidden_k"),
            )
            .with_context(|| "failed to load _dense_hidden_k")?,
            dense_q: candle_nn::linear_no_bias(
                ANTECEDENT_HIDDEN_SIZE,
                HIDDEN_SIZE,
                vb.pp("_dense_q"),
            )
            .with_context(|| "failed to load _dense_q")?,
            dense_k: candle_nn::linear_no_bias(
                ANTECEDENT_HIDDEN_SIZE,
                HIDDEN_SIZE,
                vb.pp("_dense_k"),
            )
            .with_context(|| "failed to load _dense_k")?,
        })
    }

    fn tag_logits(&self, word_embeddings: &Tensor) -> Result<Tensor> {
        let hidden = self
            .dense_hidden_tags
            .forward(word_embeddings)
            .with_context(|| "dense_hidden_tags forward failed")?
            .relu()
            .with_context(|| "tag hidden relu failed")?;

        self.dense_tags
            .forward(&hidden)
            .with_context(|| "dense_tags forward failed")
    }

    fn antecedent_scores(
        &self,
        embeddings: &Tensor,
        all_mentions: &[(usize, usize)],
        current_mentions: &[(usize, usize)],
    ) -> Result<Tensor> {
        let batched = BatchedEmbeddings::new(embeddings);
        let shared_mentions = std::ptr::eq(all_mentions, current_mentions);
        let all_embeddings = batched.gather_mention_embeddings(all_mentions)?;
        let current_embeddings = if shared_mentions {
            all_embeddings.clone()
        } else {
            batched.gather_mention_embeddings(current_mentions)?
        };

        let keys = self
            .dense_k
            .forward(
                &self
                    .dense_hidden_k
                    .forward(&all_embeddings)
                    .with_context(|| "dense_hidden_k forward failed")?
                    .relu()
                    .with_context(|| "key hidden relu failed")?,
            )
            .with_context(|| "dense_k forward failed")?;

        let queries = self
            .dense_q
            .forward(
                &self
                    .dense_hidden_q
                    .forward(&current_embeddings)
                    .with_context(|| "dense_hidden_q forward failed")?
                    .relu()
                    .with_context(|| "query hidden relu failed")?,
            )
            .with_context(|| "dense_q forward failed")?;

        let scores = queries.matmul(&keys.transpose(1, 2)?)?;
        (scores / (HIDDEN_SIZE as f64).sqrt()).with_context(|| "antecedent score scaling failed")
    }
}

struct BatchedEmbeddings<'a> {
    embeddings: &'a Tensor,
}

impl<'a> BatchedEmbeddings<'a> {
    fn new(embeddings: &'a Tensor) -> Self {
        Self { embeddings }
    }

    fn gather_word_embeddings(&self, word_indices: &[usize]) -> Result<Tensor> {
        let mut rows = Vec::with_capacity(word_indices.len().saturating_sub(1));

        for &index in &word_indices[..word_indices.len() - 1] {
            rows.push(self.flattened_position(index)?);
        }

        let refs: Vec<&Tensor> = rows.iter().collect();
        Tensor::stack(&refs, 0)?
            .unsqueeze(0)
            .with_context(|| "failed to add batch dimension to word embeddings")
    }

    fn gather_mention_embeddings(&self, mentions: &[(usize, usize)]) -> Result<Tensor> {
        let mut rows = Vec::with_capacity(mentions.len());

        for &(start, end) in mentions {
            let start_embedding = self.flattened_position(start)?;
            let end_embedding = self.flattened_position(end)?;
            rows.push(Tensor::cat(&[&start_embedding, &end_embedding], 0)?);
        }

        let refs: Vec<&Tensor> = rows.iter().collect();
        Tensor::stack(&refs, 0)?
            .unsqueeze(0)
            .with_context(|| "failed to add batch dimension to mention embeddings")
    }

    fn flattened_position(&self, index: usize) -> Result<Tensor> {
        self.embeddings
            .get(0)
            .with_context(|| "failed to select embedding batch 0")?
            .get(index)?
            .flatten_all()
            .with_context(|| format!("failed to flatten embedding at position {index}"))
    }
}

struct TensorOps;

impl TensorOps {
    fn rms_norm(input: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
        let input_dtype = input.dtype();

        let variance = input
            .sqr()
            .with_context(|| "rms_norm sqr failed")?
            .mean_keepdim(candle_core::D::Minus1)
            .with_context(|| "rms_norm mean failed")?;

        let denom = (variance + eps)?
            .sqrt()
            .with_context(|| "rms_norm sqrt failed")?
            .broadcast_as(input.shape())
            .with_context(|| "rms_norm denom broadcast failed")?;

        (input / &denom)?
            .to_dtype(input_dtype)
            .with_context(|| "rms_norm cast failed")?
            .broadcast_mul(weight)
            .with_context(|| "rms_norm weight multiply failed")
    }

    fn gelu(input: &Tensor) -> Result<Tensor> {
        input.gelu().with_context(|| "gelu fused op failed")
    }

    fn linear_no_bias_forward(input: &Tensor, weight: &Tensor) -> Result<Tensor> {
        let input_shape = input.dims().to_vec();
        anyhow::ensure!(
            !input_shape.is_empty(),
            "linear_no_bias_forward expects at least 1 dimension",
        );

        let in_dim = *input_shape.last().unwrap();
        let weight_shape = weight.dims();

        anyhow::ensure!(
            weight_shape.len() == 2,
            "weight must be rank 2, got {:?}",
            weight_shape
        );
        anyhow::ensure!(
            weight_shape[1] == in_dim,
            "input dim mismatch: x last dim={}, weight={:?}",
            in_dim,
            weight_shape,
        );

        let leading: usize = input_shape[..input_shape.len() - 1].iter().product();
        let out_dim = weight_shape[0];

        let input_2d = input
            .reshape((leading, in_dim))
            .with_context(|| "linear_no_bias reshape to 2d failed")?;
        let output_2d = input_2d
            .matmul(&weight.t()?)
            .with_context(|| "linear_no_bias matmul failed")?;

        let mut output_shape = input_shape;
        *output_shape.last_mut().unwrap() = out_dim;

        output_2d
            .reshape(output_shape)
            .with_context(|| "linear_no_bias reshape output failed")
    }

    fn softmax_last_dim(input: &Tensor) -> Result<Tensor> {
        candle_nn::ops::softmax(input, candle_core::D::Minus1).with_context(|| "softmax failed")
    }

    fn split_heads(input: &Tensor, num_heads: usize, head_dim: usize) -> Result<Tensor> {
        let dims = input.dims();
        anyhow::ensure!(
            dims.len() == 3,
            "split_heads expected rank 3, got {:?}",
            dims
        );

        input
            .reshape((dims[0], dims[1], num_heads, head_dim))
            .with_context(|| "split_heads reshape failed")?
            .permute((0, 2, 1, 3))
            .with_context(|| "split_heads permute failed")
    }

    fn merge_heads(input: &Tensor) -> Result<Tensor> {
        let dims = input.dims();
        anyhow::ensure!(
            dims.len() == 4,
            "merge_heads expected rank 4, got {:?}",
            dims
        );

        input
            .permute((0, 2, 1, 3))
            .with_context(|| "merge_heads permute failed")?
            .reshape((dims[0], dims[2], dims[1] * dims[3]))
            .with_context(|| "merge_heads reshape failed")
    }
}

#[cfg(test)]
mod tests {
    use super::SelfAttentionBlock;

    #[test]
    fn relative_position_bucket_is_stable_for_known_values() {
        assert_eq!(SelfAttentionBlock::relative_position_bucket(0), 0);
        assert_eq!(SelfAttentionBlock::relative_position_bucket(-1), 17);
        assert_eq!(SelfAttentionBlock::relative_position_bucket(1), 1);
    }
}
