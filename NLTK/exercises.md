Exercises
===========

1. Try using the Python interpreter as a calculator, and typing expressions like `12 / (4 + 1)`.
2. Given an alphabet of 26 letters, there are 26 to the power 10, or `26 ** 10`, ten-letter strings we can form. That works out to 141167095653376. How many hundred-letter strings are possible?
3. The Python multiplication operation can be applied to lists. What happens when you type `['Monty', 'Python'] * 20`, or `3 * sent1`?
4. How many words are there in `text2`? How many distinct words are there?
5. Compare the lexical diversity scores for humor and romance fiction in 1.1. Which genre is more lexically diverse?
6. Produce a dispersion plot of the four main protagonists in `Sense and Sensibility`: `Elinor`, `Marianne`, `Edward`, and `Willoughby`. What can you observe about the different roles played by the males and females in this novel? Can you identify the couples?
7. Find the collocations in text5.
8. Consider the following Python expression: `len(set(text4))`. State the purpose of this expression. Describe the two steps involved in performing this computation.
9. Define a string and assign it to a variable, e.g., `my_string = 'My String'` (but put something more interesting in the string). 
    - Print the contents of this variable in two ways, first by simply typing the variable name and pressing enter, then by using the print statement.
    - Try adding the string to itself using `my_string + my_string`, or multiplying it by a number, e.g.,`my_string * 3`. 
    - Notice that the strings are joined together without any spaces. How could you fix this?
10. Define a variable `my_sent` to be a list of words, using the syntax `my_sent = ["My", "sent"]` (but with your own words, or a favorite saying).
    - Use `' '.join(my_sent)` to convert this into a string.
    - Use split() to split the string back into the list form you had to start with.
11. Define several variables containing lists of words, e.g., `phrase1`, `phrase2`, and so on. Join them together in various combinations (using the plus operator) to form whole sentences. What is the relationship between `len(phrase1 + phrase2)` and `len(phrase1) + len(phrase2)`?
12. Consider the following two expressions, which have the same value. Which one will typically be more relevant in NLP? Why?
    ```
    "Monty Python"[6:12]
    ["Monty", "Python"][1]
    ```
13. We have seen how to represent a sentence as a list of words, where each word is a sequence of characters. What does `sent1[2][2]` do? Why? Experiment with other index values.
14. The first sentence of `text3` is provided to you in the variable sent3. The index of `the` in `sent3` is 1, because `sent3[1]` gives us 'the'. What are the indexes of the two other occurrences of this word in `sent3`?
15. Find all words in the `Chat Corpus` (`text5`) starting with the letter `b`. Show them in alphabetical order.
16. Type the expression `list(range(10))` at the interpreter prompt. Now try `list(range(10, 20))`, `list(range(10, 20, 2))`, and `list(range(20, 10, -2))`. We will see a variety of uses for this built-in function in later part of the unit.
17. Use `text9.index()` to find the index of the word `sunset`. You'll need to insert this word as an argument between the parentheses. By a process of trial and error, find the slice for the complete sentence that contains this word.
    - Using list addition, and the set and sorted operations, compute the vocabulary of the sentences `sent1 ... sent8`.
18. What is the difference between the following two lines? Which one will give a larger value? Will this be the case for other texts?
    ```
    sorted(set(w.lower() for w in text1))
    sorted(w.lower() for w in set(text1))
    ```
19. What is the difference between the following two tests: `w.isupper()` and `not w.islower()`?
20. Write the slice expression that extracts the last two words of `text2`.
21. Find all the four-letter words in `the Chat Corpus` (`text5`). With the help of a frequency distribution (FreqDist), show these words in decreasing order of frequency.
22. Review the discussion of looping with conditions. Use a combination of `for` and `if` statements to loop over the words of the movie script for `Monty Python and the Holy Grail` (`text6`) and print all the uppercase words, one per line.
23. Write expressions for finding all words in `text6` that meet the conditions listed below. The result should be in the form of a list of words: `['word1', 'word2', ...]`.
    - Ending in ise
    - Containing the letter z
    - Containing the sequence of letters pt
    - Having all lowercase letters except for an initial capital (i.e., titlecase)
24. Define sent to be the list of words ['she', 'sells', 'sea', 'shells', 'by', 'the', 'sea', 'shore']. Now write code to perform the following tasks:
    - Print all words beginning with sh
    - Print all words longer than four characters
25. What does the following Python code do? `sum(len(w) for w in text1)` Can you use it to work out the average word length of a text?
26. Define a function called `vocab_size(text)` that has a single parameter for the text, and which returns the vocabulary size of the text.
27. Define a function percent(word, text) that calculates how often a given word occurs in a text, and expresses the result as a percentage.
28. We have been using sets to store vocabularies. Try the following Python expression: `set(sent3) < set(text1)`. Experiment with this using different arguments to `set()`. What does it do? Can you think of a practical application for this?