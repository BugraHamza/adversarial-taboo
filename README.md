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

## Mind Map

Papers and Researches

id | Name | Release Date | Link
---|------|--------------|----------------
1 | Adversarial Language Games for Advanced Natural Language Intelligence | 17 Dec 2020 | https://arxiv.org/pdf/1911.01622.pdf 
2 | Unsupervised Question Answering by Cloze Translation | 27 Jun 2019 | https://arxiv.org/pdf/1906.04980.pdf
-1 | Adversarial Ranking for Language Generation | 16 Apr 2018 | https://arxiv.org/pdf/1705.11001.pdf
-1 | Boot-strapping a neural conversational agent with dialogue self-play | 15 Jan 2018 | https://www.researchgate.net/publication/322518246_Building_a_Conversational_Agent_Overnight_with_Dialogue_Self-Play/link/5b6fcd2e45851546c9fb91d9/download
-1 | Building end-to-end dialogue systems using generative hierarchical neural network models | 6 Apr 2016 | https://arxiv.org/pdf/1507.04808.pdf
-1 | Deep Reinforcement Learning for Dialogue Generation | 29 Sep 2016 | https://arxiv.org/pdf/1606.01541.pdf

## Requirements

1. Dataset Collection
- Wikipedia datasets
- Reddit Conversation dataset
- Forum DonanımHaber dataset

2. Word Selection 
- Word Selection using Zemberek (Java)

3. Judge System
- Fine-tune GPT2 for fluency check (Currently works well)
- Fine-tune BERT for relevancy check (Stuck at 0.75 accuracy, need to be improved in the future)

4. OpenQA-Based Simulation
- Neural Question Generation model for attacker [2]
- Context-based question answering using BERT for defender
- Context based question answering using DocQA for defender

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
    