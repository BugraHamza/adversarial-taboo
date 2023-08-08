# AutoTaboo Player

AI plays Taboo games

The language is an ordinary tool for humans. In recent years with the improvement in AI, computers can also make sentences, but a machine has to understand and answer the understood question for a conversation as well. For machines it is still a big challenge in order to answer question, and therefore to communicate human-like. Dialog generation is an subfield of NLP. Sentence generation is a task for dialog generation, but the followings have to be taken into account as well:
- Relevancy
- Fluency
- Diversity

Pragmatics for short.

These properties were less promoted up until now, but are essential to communicate in a context. 

## Game Definitions
Classic(Cooperative) Taboo: In this taboo version, teller takes a card with a word to tell and a bunch of (Restricted) words not to use.

Adversarial Taboo: In this version of taboo, there exists an attacker and a defender. Attacker tells a word and defender tries to find out which word the attacker tends to. If the defender finds the word, it wins. If the defender tells the word, it loses. If the word isn't found in the given time, it is a draw.

## Requirements

1. Dataset Collection
- Wikipedia datasets
- Reddit Conversation dataset
- Forum DonanımHaber dataset

2. Word Selection 
- Word Selection using Zemberek (Java)

3. Judge System
- Fine-tune GPT2 for fluency check 
- Fine-tune BERT for relevancy check

4. OpenQA-Based Simulation
- Neural Question Generation model for attacker
- Context-based question answering using BERT for defender
- Context based question answering using DocQA for defender
    