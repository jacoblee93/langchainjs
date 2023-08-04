import { BaseTransformOutputParser } from "../schema/output_parser.js";

export class NoOpOutputParser<T> extends BaseTransformOutputParser<T> {
  lc_namespace = ["langchain", "output_parsers", "default"];

  lc_serializable = true;

  parse(text: string): Promise<T> {
    return Promise.resolve(text as T);
  }

  getFormatInstructions(): string {
    return "";
  }
}
