package com.wordselection;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;
import zemberek.morphology.TurkishMorphology;
import zemberek.tokenization.TurkishSentenceExtractor;
import zemberek.langid.LanguageIdentifier;

import java.io.*;
import java.util.*;

public class Main {
    public static void main(String[] args) throws IOException {
        final String inFile = "/Users/bugrahamzagundog/Desktop/AutoTaboo Player/datasets/Turkish-Wiki-Dataset/tr_wiki.csv";
        final String outFile = "wiki_word_selection.txt";

        Counter counter = new Counter();
        TurkishMorphology morphology = TurkishMorphology.createWithDefaults();
        TurkishSentenceExtractor extractor = TurkishSentenceExtractor.DEFAULT;
        LanguageIdentifier identifier = LanguageIdentifier.fromInternalModels();

        int rowCount = 0;
        CSVParser parser = new CSVParser(new FileReader(inFile), CSVFormat.RFC4180);
        for(final CSVRecord record : parser) {
            System.out.printf("%d\r", rowCount++);
            // Get content
            String content = record.get(2);

            // Analyze content
            List<String> sentences = extractor.fromParagraph(content);
            for(String sent : sentences) {
                morphology.analyzeAndDisambiguate(sent).bestAnalysis().forEach(s -> {
                    String lemma = s.getLemmas().get(0);
                    String pos = s.getPos().getStringForm();
                    Word w = new Word(lemma, pos);
                    counter.addWord(w);
                });
            }
        }

        FileWriter writer = new FileWriter(outFile);

        HashMap<Word, Integer> sorted = counter.sortedCounter();
        for(Map.Entry<Word, Integer> entry : sorted.entrySet()) {
            Word key = entry.getKey();
            int count = entry.getValue();

            String word = key.getWord();
            String pos = key.getPosTag();

            // if pos is Noun or the word contains only alphabet characters
            if(pos.equals("Noun") && word.matches("[a-zA-Z]+") && word.length() > 2 && identifier.identify(word).equals("tr")) {
                writer.write(word + " " + count + "\n");
               System.out.println(word + " " + count);
            }
        }

        writer.close();
    }
}
