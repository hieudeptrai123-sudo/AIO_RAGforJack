from torch.utils.data import Dataset
class LegalDataset(Dataset):
    def __init__(self, df, tokenizer_question, tokenizer_context, max_length):
        self.df = df
        self.tokenizer_question = tokenizer_question
        self.tokenizer_context = tokenizer_context
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        question = self.df.iloc[idx]['question']
        context = self.df.iloc[idx]['context']

        # Ensure question and context are strings
        if not isinstance(question, str):
            print(f"Non-string question at index {idx}: {question} (type: {type(question)})")
            question = str(question)  # Convert to string if necessary

        if not isinstance(context, str):
            print(f"Non-string context at index {idx}: {context} (type: {type(context)})")
            context = str(context)  # Convert to string if necessary

        # Tokenize the question and context
        question_encoding = self.tokenizer_question(
            question,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        context_encoding = self.tokenizer_context(
            context,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': question_encoding['input_ids'].squeeze(),
            'attention_mask': question_encoding['attention_mask'].squeeze(),
            'context_input_ids': context_encoding['input_ids'].squeeze(),
            'context_attention_mask': context_encoding['attention_mask'].squeeze()
        }
