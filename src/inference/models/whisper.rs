use anyhow::{Error, Result};
use hf_hub::api::sync::Api;
use hf_hub::{Repo, RepoType};
use rand::SeedableRng;

use crate::inference::audio_pipeline::AudioGeneratorPipeline;
use crate::inference::models::model::ModelBase;
use crate::inference::task::transcribe::{TranscribeHandler, TranscribeResponse};

// Taken from https://github.com/huggingface/candle/blob/main/candle-examples/examples/whisper/main.rs
#[derive(Clone)]
pub struct WhisperModel {
    pub base: ModelBase,
    generator_pipeline: AudioGeneratorPipeline,
}

impl WhisperModel {
    #[tracing::instrument(level = "info", skip(api))]
    pub fn new(
        api: Api,
        base: &ModelBase,
    ) -> Result<Self> {
        let repo = api.repo(Repo::with_revision(
            base.repo_id.clone(),
            RepoType::Model,
            base.repo_revision.clone(),
        ));
        let generator_pipeline = AudioGeneratorPipeline::with_model(
            &repo,
            true,
            rand::rngs::StdRng::from_seed([0; 32]),
        )?;

        Ok(Self {
            base: base.clone(),
            generator_pipeline,
        })
    }
}

impl TranscribeHandler for WhisperModel {
    #[tracing::instrument(level = "info", skip(self, input))]
    fn run_transcribe(
        &mut self,
        input: Box<[u8]>,
        language_token: &str,
    ) -> Result<TranscribeResponse, Error> {
        //start timer for inference time
        let start = std::time::Instant::now();
        let output = self.generator_pipeline.transcribe(input, language_token)?;
        let inference_time = start.elapsed().as_secs_f64();
        Ok(TranscribeResponse {
            output,
            inference_time:inference_time,
        })
    }
}
