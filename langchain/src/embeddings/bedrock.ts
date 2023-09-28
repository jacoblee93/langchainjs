import { SignatureV4 } from "@smithy/signature-v4";
import { defaultProvider } from "@aws-sdk/credential-provider-node";
import { HttpRequest } from "@smithy/protocol-http";
import { Sha256 } from "@aws-crypto/sha256-js";

import { Embeddings, EmbeddingsParams } from "./base.js";
import { type CredentialType } from "../util/bedrock.js";
import { getEnvironmentVariable } from "../util/env.js";

/**
 * Interface for the configuration options of the BedrockEmbeddings
 * class.
 */
export interface BedrockEmbeddingsConfig extends EmbeddingsParams {
  /** The AWS region to use. */
  region?: string;

  /** The ID of the Bedrock model to use. */
  model: string;

  /** AWS Credentials.
      If no credentials are provided, the default credentials from `@aws-sdk/credential-provider-node` will be used.
   */
  credentials?: CredentialType;

  /** Key word arguments to pass to the model. */
  modelKwargs?: Record<string, unknown>;

  /** Override the default endpoint hostname. */
  endpointHost?: string;
}

/**
 * Class that extends the BaseEmbeddings class and implements the
 * Embeddings interface. It provides methods for embedding text using the
 * Bedrock service.
 */
export class BedrockEmbeddings
  extends Embeddings
  implements BedrockEmbeddingsConfig
{
  static lc_name() {
    return "BedrockEmbeddings";
  }

  region: string;

  model: string;

  credentials: CredentialType;

  endpointHost?: string;

  modelKwargs?: Record<string, unknown>;

  constructor(fields: BedrockEmbeddingsConfig) {
    super(fields);
    const region =
      fields?.region ?? getEnvironmentVariable("AWS_DEFAULT_REGION");
    if (!region) {
      throw new Error(
        "Please set the AWS_DEFAULT_REGION environment variable or pass it to the constructor as the region field."
      );
    }
    this.region = region;
    this.credentials = fields?.credentials ?? defaultProvider();
    this.endpointHost = fields?.endpointHost;
    this.modelKwargs = fields?.modelKwargs;
  }

  protected async _embed(text: string): Promise<number[]> {
    const service = "bedrock-runtime";

    const endpointHost =
      this.endpointHost ?? `${service}.${this.region}.amazonaws.com`;

    const url = new URL(`https://${endpointHost}/model/${this.model}/invoke`);

    const inputBody = {
      ...this.modelKwargs,
      inputText: text,
    };

    const request = new HttpRequest({
      hostname: url.hostname,
      path: url.pathname,
      protocol: url.protocol,
      method: "POST", // method must be uppercase
      body: JSON.stringify(inputBody),
      query: Object.fromEntries(url.searchParams.entries()),
      headers: {
        // host is required by AWS Signature V4: https://docs.aws.amazon.com/general/latest/gr/sigv4-create-canonical-request.html
        host: url.host,
        accept: "application/json",
        "content-type": "application/json",
      },
    });

    const signer = new SignatureV4({
      credentials: this.credentials,
      service: "bedrock",
      region: this.region,
      sha256: Sha256,
    });

    const signedRequest = await signer.sign(request);
    const response = await this.caller.call(async () =>
      fetch(url, {
        headers: signedRequest.headers,
        body: signedRequest.body,
        method: signedRequest.method,
      })
    );
    console.log(await response.json());
    return [];
  }

  async embedQuery(text: string): Promise<number[]> {
    return this._embed(text);
  }

  async embedDocuments(texts: string[]): Promise<number[][]> {
    return Promise.all(texts.map((text) => this._embed(text)));
  }
}
