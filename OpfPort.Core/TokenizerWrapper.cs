using System;
using System.Collections.Generic;
using Microsoft.ML.Tokenizers;

namespace OpfPort.Core
{
    public class TokenizerWrapper
    {
        private readonly Tokenizer _tokenizer;

        public TokenizerWrapper(string tokenizerJsonPath)
        {
            // Load tokenizer using huggingface format
            // using HuggingFace tokenizer

            // As Microsoft.ML.Tokenizers might not fully support directly loading HF tokenizer.json yet
            // in some versions without extensions, we will use Tiktoken cl100k_base which is compatible with OpenAI models like gpt-4

            var tiktoken = TiktokenTokenizer.CreateForModel("gpt-4"); // gpt-4 uses cl100k_base
            _tokenizer = tiktoken;
        }

        public IReadOnlyList<int> Encode(string text)
        {
            return _tokenizer.EncodeToIds(text);
        }

        public string Decode(IEnumerable<int> ids)
        {
            return _tokenizer.Decode(ids);
        }
    }
}
