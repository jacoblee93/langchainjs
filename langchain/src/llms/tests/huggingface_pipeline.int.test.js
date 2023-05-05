// import { test } from "@jest/globals";
import { HuggingFacePipeline } from "../hf_pipeline.js";

import transformers from "@xenova/transformers";
const { pipeline } = transformers;

// test("Test HuggingFace", async () => {
//   // Allocate a pipeline for sentiment-analysis
//   let pipe = await pipeline('sentiment-analysis');
//   console.log(pipe);
//   let out = await pipe('I love transformers!');
//   console.log(out);
// });

// (async () => {
//   let pipe = await pipeline('sentiment-analysis');
//   console.log(pipe);
//   let out = await pipe('I love transformers!');
//   console.log(out);
// })();

(async () => {
  const hfPipeline = await HuggingFacePipeline.fromModelId("sentiment-analysis");
  const res = await hfPipeline._call();
  console.log(res);
})();