import { test, expect } from "@jest/globals";
import { GPT4All } from "../gpt4all.js";

// GPT4All will likely need to download the model, which may take a couple mins
test(
  "Test GPT4All",
  async () => {
    const startTime = performance.now();
    const model = new GPT4All({
      model: "gpt4all-lora-quantized",
    });

    const res = await model.call("Hello, my name is ");

    const endTime = performance.now();
    const timeElapsed = endTime - startTime;
    console.log(`GPT4All: Time elapsed: ${timeElapsed} milliseconds`);

    expect(typeof res).toBe("string");
  },
  600 * 1000
);

test(
  "Test GPT4All with multiple concurrent uninitialized calls",
  async () => {
    const startTime = performance.now();
    const model = new GPT4All({
      model: "gpt4all-lora-quantized",
    });

    const responses = await Promise.all([
      model.call("Hello, my name is "),
      model.call("What color is the sky?"),
      model.call("Who are you?"),
    ]);
    const endTime = performance.now();
    const timeElapsed = endTime - startTime;
    console.log(`GPT4All: Time elapsed: ${timeElapsed} milliseconds`);

    console.log({ responses });
  },
  600 * 1000
);
