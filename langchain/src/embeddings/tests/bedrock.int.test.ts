/* eslint-disable no-process-env */
/* eslint-disable @typescript-eslint/no-non-null-assertion */

import { test, expect } from "@jest/globals";
import { BedrockEmbeddings } from "../bedrock.js";

test.only("Test BedrockEmbeddings.embedQuery", async () => {
  const embeddings = new BedrockEmbeddings({
    model: "amazon.titan-embed-text-v1",
    region: "us-east-1",
    credentials: {
      accessKeyId: process.env.BEDROCK_AWS_ACCESS_KEY_ID!,
      secretAccessKey: process.env.BEDROCK_AWS_SECRET_ACCESS_KEY!,
    },
  });
  const res = await embeddings.embedQuery("Hello world");
  expect(typeof res[0]).toBe("number");
});

test("Test BedrockEmbeddings.embedDocuments", async () => {
  const embeddings = new BedrockEmbeddings({
    model: "amazon.titan-embed-text-v1",
    region: "us-east-1",
    credentials: {
      accessKeyId: process.env.BEDROCK_AWS_ACCESS_KEY_ID!,
      secretAccessKey: process.env.BEDROCK_AWS_SECRET_ACCESS_KEY!,
    },
  });
  const res = await embeddings.embedDocuments(["Hello world", "Bye bye"]);
  expect(res).toHaveLength(2);
  expect(typeof res[0][0]).toBe("number");
  expect(typeof res[1][0]).toBe("number");
});
