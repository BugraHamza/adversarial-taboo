package com.wordselection;

import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.stream.Collectors;

public class Counter {
    private HashMap<Word, Integer> counter;

    public Counter() {
        this.counter = new HashMap<>();
    }

    public HashMap<Word, Integer> getCounter() {
        return counter;
    }

    public void setCounter(HashMap<Word, Integer> counter) {
        this.counter = counter;
    }

    public void addWord(Word word) {
        Integer wordCount = this.counter.get(word);
        if(wordCount == null)
            this.counter.put(word, 1);
        else
            this.counter.replace(word, wordCount, wordCount+1);
    }

    public HashMap<Word, Integer> sortedCounter() {
        HashMap<Word, Integer> sortedMap = this.counter.entrySet()
                .stream().sorted(Comparator.comparing(Map.Entry::getValue))
                .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue,
                        (e1, e2) -> e1, LinkedHashMap::new));
        return sortedMap;
    }

    @Override
    public String toString() {
        return "" + counter;
    }
}
