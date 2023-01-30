package com.wordselection;

import java.util.Objects;

public class Word {
    private String word;
    private String posTag;

    public Word(String word, String posTag) {
        this.word = word;
        this.posTag = posTag;
    }

    public String getWord() {
        return this.word;
    }

    public void setWord(String newWord) {
        this.word = newWord;
    }

    public String getPosTag() {
        return this.posTag;
    }

    public void setPosTag(String newPosTag) {
        this.posTag = newPosTag;
    }


    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof Word)) return false;
        Word word1 = (Word) o;
        return word.equals(word1.word) && posTag.equals(word1.posTag);
    }

    @Override
    public int hashCode() {
        return Objects.hash(word, posTag);
    }

    @Override
    public String toString() {
        return "(" + word + ", " + posTag + ')';
    }
}
