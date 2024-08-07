import { test, describe, expect } from "@jest/globals";
import { AlephAlpha } from "../aleph_alpha.js";

describe("Aleph Alpha", () => {
  test("test call", async () => {
    const aleph_alpha = new AlephAlpha({});
    // @eslint-disable-next-line/@typescript-eslint/ban-ts-comment
    // @ts-expect-error unused var
    const result = await aleph_alpha.invoke(
      "What is a good name for a company that makes colorful socks?"
    );
    // console.log({ result });
  });

  test("test translation call", async () => {
    const aleph_alpha = new AlephAlpha({});
    // @eslint-disable-next-line/@typescript-eslint/ban-ts-comment
    // @ts-expect-error unused var
    const result = await aleph_alpha.invoke(
      `Translate "I love programming" into German.`
    );
    // console.log({ result });
  });

  test("test JSON output call", async () => {
    const aleph_alpha = new AlephAlpha({});
    // @eslint-disable-next-line/@typescript-eslint/ban-ts-comment
    // @ts-expect-error unused var
    const result = await aleph_alpha.invoke(
      `Output a JSON object with three string fields: "name", "birthplace", "bio".`
    );
    // console.log({ result });
  });

  test("should abort the request", async () => {
    const aleph_alpha = new AlephAlpha({});
    const controller = new AbortController();

    await expect(() => {
      const ret = aleph_alpha.invoke(
        "Respond with an extremely verbose response",
        {
          signal: controller.signal,
        }
      );
      controller.abort();
      return ret;
    }).rejects.toThrow("AbortError: This operation was aborted");
  });

  test("throws an error when response status is not ok", async () => {
    const aleph_alpha = new AlephAlpha({
      aleph_alpha_api_key: "BAD_KEY",
    });

    await expect(aleph_alpha.invoke("Test prompt")).rejects.toThrow(
      'Aleph Alpha call failed with status 401 and body {"error":"InvalidToken","code":"UNAUTHENTICATED"}'
    );
  });
});
