package com.test.ocrdemo;

import java.util.HashMap;
import java.util.Map;

public class OcrStrLabelConverter {
    private final char[] alphabet;

    public OcrStrLabelConverter(String alphabets) {
        this.alphabet = alphabets.toCharArray();
    }

    public String decode(int[] t) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < t.length; i++) {
            if (t[i] != 0 && !(i > 0 && t[i - 1] == t[i])) {
                sb.append(this.alphabet[t[i] - 1]);
            }
        }
        return sb.toString();
    }
}
