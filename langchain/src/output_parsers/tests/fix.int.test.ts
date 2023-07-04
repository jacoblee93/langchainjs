import { z } from "zod";
import { BaseChatModel } from "../../chat_models/base.js";
import { BaseMessage, ChatResult } from "../../schema/index.js";
import { LLMChain } from "../../chains/llm_chain.js";
import { OutputFixingParser } from "../fix.js";
import { FunctionCallStructuredOutputParser } from "../openai_function_call_structured.js";
import {
  ChatPromptTemplate,
  HumanMessagePromptTemplate,
  SystemMessagePromptTemplate,
} from "../../prompts/index.js";
import { ChatOpenAI } from "../../chat_models/openai.js";
import { AIMessage } from "../../schema/index.js";

class FakeChatModel extends BaseChatModel {
  nrMapCalls = 0;

  nrReduceCalls = 0;

  _llmType(): string {
    return "fake";
  }

  _combineLLMOutput(..._llmOutputs: (Record<string, any> | undefined)[]) {
    return undefined;
  }

  async _generate(
    messages: BaseMessage[],
    _options: this["ParsedCallOptions"]
  ): Promise<ChatResult> {
    return {
      generations: messages.map((_message) => {
        return {
          text: "",
          message: new AIMessage({
            content: "",
            additional_kwargs: {
              function_call: {
                name: "__lc_output__",
                arguments: JSON.stringify({
                  name: {
                    value: "Adam",
                  },
                  surname: {
                    value: "Nowak",
                  },
                  age: {
                    value: "44",
                  },
                  appearance: {
                    value: "Tall with dark hair",
                  },
                  shortBio: {
                    value: "A person who enjoys doing human activities",
                  },
                  university: {
                    value: "Warsaw University",
                  },
                  gender: {
                    value: "male",
                  },
                  interests: {
                    value: [
                      {
                        value: "Skiing",
                      },
                      {
                        value: "Video Games",
                      },
                    ],
                  },
                }),
              },
            },
          }),
        };
      }),
    };
  }
}

test("Test OutputFixingParser with a FunctionCallStructuredOutputParser", async () => {
  const fakeChatModel = new FakeChatModel({});
  const realChatModel = new ChatOpenAI({
    modelName: "gpt-3.5-turbo-0613",
    temperature: 0,
  });

  const baseParser = FunctionCallStructuredOutputParser.fromZodSchema(
    z.object({
      name: z.string().describe("Human name"),
      surname: z.string().describe("Human surname"),
      age: z.number().describe("Human age"),
      appearance: z.string().describe("Human appearance description"),
      shortBio: z.string().describe("Short bio secription"),
      university: z.string().optional().describe("University name if attended"),
      gender: z.string().describe("Gender of the human"),
      interests: z
        .array(z.string())
        .describe("json array of strings human interests"),
    })
  );

  const fixingParser = OutputFixingParser.fromLLM(realChatModel, baseParser);
  const prompt = new ChatPromptTemplate({
    promptMessages: [
      SystemMessagePromptTemplate.fromTemplate(
        "Generate details of a hypothetical person."
      ),
      HumanMessagePromptTemplate.fromTemplate(
        "Person description: {inputText}"
      ),
    ],
    inputVariables: ["inputText"],
  });

  const chain = new LLMChain({
    llm: fakeChatModel,
    prompt,
    outputParser: fixingParser,
  });

  const response = await chain.call({ inputText: "A man, living in Poland." });
  console.log("response", response);

  expect(response.person).toHaveProperty("name");
  expect(response.person).toHaveProperty("surname");
  expect(response.person).toHaveProperty("age");
  expect(response.person).toHaveProperty("appearance");
  expect(response.person).toHaveProperty("shortBio");
  expect(response.person).toHaveProperty("age");
  expect(response.person).toHaveProperty("gender");
  expect(response.person).toHaveProperty("interests");
  expect(response.person.interests.length).toBeGreaterThan(0);
});
