# AutoTaboo Player

AI plays Taboo games

The language is an ordinary tool for humans. In recent years with the improvement in AI, computers can also make sentences, but a machine has to understand and answer the understood question for a conversation as well. For machines it is still a big challenge in order to answer question, and therefore to communicate human-like. Dialog generation is an subfield of NLP. Sentence generation is a task for dialog generation, but the followings have to be taken into account as well:
- Adequacy
- Fluency
- Diversity

Pragmatics for short.

These properties were less promoted up till now, but are essential to communicate in a context. 

## Game Definitions
Classic(Cooperative) Taboo: In this taboo version, teller takes a card with a word to tell and a bunch of words not to use.

Adversarial Taboo: In this version of taboo, there exists an attacker and a defender. Attacker tells a word and defender tries to find out which word the attacker tends to. If the defender finds the word, it wins. If the defender tells the word, it loses.

## Mind Map

Papers and Researches

id | Name | Release Date | Link
---|------|--------------|----------------
1 | Adversarial Language Games for Advanced Natural Language Intelligence | 17 Dec 2020 | https://arxiv.org/pdf/1911.01622.pdf 
2 | Adversarial Ranking for Language Generation | 16 Apr 2018 | https://arxiv.org/pdf/1705.11001.pdf
3 | Boot-strapping a neural conversational agent with dialogue self-play | 15 Jan 2018 | https://www.researchgate.net/publication/322518246_Building_a_Conversational_Agent_Overnight_with_Dialogue_Self-Play/link/5b6fcd2e45851546c9fb91d9/download
4 | Building end-to-end dialogue systems using generative hierarchical neural network models | 6 Apr 2016 | https://arxiv.org/pdf/1507.04808.pdf
5 | Deep Reinforcement Learning for Dialogue Generation | 29 Sep 2016 | https://arxiv.org/pdf/1606.01541.pdf

## Requirements

1. Dataset Collection (Due Date: 13/02/2022)
- Wikipedia datasets (Tr/En)
- Reddit Conversation dataset (En) (Can be found in tfds)
- Forum DonanımHaber dataset (will be crawled)
- Convert each dataset to Huggingface dataset format

2. Word Selection 
- Find PoS Taggers for Turkish and English
- Using a PoS Taggers (acc > 90%), tag datasets
- Based on their speed, tagging strategy may be changed
- Select nouns with high frequency from each dataset


3. Judge System
- Fine-tune GPT2 for fluency check (Tr/En)
- Fine-tune BERT for relevancy check (Tr/En)

4. OpenQA-Based Simulation
- 

5. Chatbot-Based Simulation
- 

6. Implementing Game Environment
- 

### Adversarial Language Games for Advanced Natural Language Intelligence

Purpose : Providing solution to adversarial taboo, which is proposed in the paper itself.

Summary : We propose Turkish version to Adversarial Taboo game[1]. We implemented the strategies described in the game's paper. 

Target Words Selection -> 563 words from Wiki for OpenQA-based
                          567 words from Reddit for Chatbot-based
                          
Judge System: 
    Fluency check -> Finetuned pretrained GPT2
    Relevancy check (post/response) -> BERT
    