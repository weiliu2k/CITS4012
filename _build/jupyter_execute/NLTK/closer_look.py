A Closer Look at Python: Texts as Lists of Words
================================================

## Lists
What is a text? At one level, it is a sequence of symbols on a page such as this one. At another level, it is a sequence of chapters, made up of a sequence of sections, where each section is a sequence of paragraphs, and so on. However, for our purposes, we will think of a text as nothing more than a sequence of words and punctuation. Here's how we represent text in Python, in this case the opening sentence of `Moby Dick`:

sent1 = ['Call', 'me', 'Ishmael', '.']

Here we have given a variable name we made up, `sent1`, followed by the equals sign, and then some quoted words, separated with commas, and surrounded with a pair of square brackets. This square bracketed, comma seperated content is known as a *list* in Python: it is how we store a text. We can inspect it, ask for its length and apply our own lexical_diversity() function to it.

sent1

len(sent1)

To use the functions we defined in another Jupyter Notebook, the best practice is to create a plain python (.py) to host all the functions, rather than import the ipython file through packages like [nbimporter](https://github.com/grst/nbimporter). 

After copying the two functions into a file called `utils.py`, we can import the functions and use them in the current notebook. 

from utils import lexical_diversity
lexical_diversity(sent1)

Some more lists have been defined for you, one for the opening sentence of each of our texts, sent2 … sent9. We inspect two of them here. 

from nltk.book import *
sent2 

sent3

```{admonition} Your Turn
Make up a few sentences of your own, by typing a name, equals sign, and a list of words, like this: `ex1 = ['Monty', 'Python', 'and', 'the', 'Holy', 'Grail']`. Repeat some of the other Python operations we saw earlier in 1, e.g., `sorted(ex1)`, `len(set(ex1))`, `ex1.count('the')`.
```	

A pleasant surprise is that we can use Python's addition operator on lists. Adding two lists creates a new list with everything from the first list, followed by everything from the second list:

['Monty', 'Python'] + ['and', 'the', 'Holy', 'Grail']

This special use of the addition operation is called concatenation; it combines the lists together into a single list. We can concatenate sentences to build up a text.

We don't have to literally type the lists either; we can use short names that refer to pre-defined lists.

sent4 + sent1

What if we want to add a single item to a list? This is known as appending. When we `append()` to a list, the list itself is updated as a result of the operation.

sent1.append("Some")
sent1

## Indexing Lists

As we have seen, a text in Python is a list of words, represented using a combination of brackets and quotes. Just as with an ordinary page of text, we can count up the total number of words in text1 with `len(text1)`, and count the occurrences in a text of a particular word — say, `'heaven'` — using `text1.count('heaven')`.

With some patience, we can pick out the 1st, 173rd, or even 14,278th word in a printed text. Analogously, we can identify the elements of a Python list by their order of occurrence in the list. The number that represents this position is the item's index. We instruct Python to show us the item that occurs at an index such as 173 in a text by writing the name of the text followed by the index inside square brackets:

text4[173]

We can do the converse; given a word, find the index of *when it first occurs*:

text4.index('awaken')

Indexes are a common way to access the words of a text, or, more generally, the elements of any list. Python permits us to access sublists as well, extracting manageable pieces of language from large texts, a technique known as *slicing*.

text5[16715:16735]

Indexes have some subtleties, and we'll explore these with the help of an artificial sentence:

sent = ['word1', 'word2', 'word3', 'word4', 'word5',
...         'word6', 'word7', 'word8', 'word9', 'word10']
sent[0]

sent[9]

```{caution}
Notice that our indexes start from zero: sent element zero, written sent[0], is the first word, 'word1', whereas sent element 9 is 'word10'. The reason is simple: the moment Python accesses the content of a list from the computer's memory, it is already at the first element; we have to tell it how many elements forward to go. Thus, zero steps forward leaves it at the first element.
```

This practice of counting from zero is initially confusing, but typical of modern programming languages. You'll quickly get the hang of it if you've mastered the system of counting centuries where 19XY is a year in the 20th century, or if you live in a country where the floors of a building are numbered from 1, and so walking up n-1 flights of stairs takes you to level n.

if we accidentally use an index that is too large, we get an error:

sent[10]

This time it is *not a syntax error*, because the program fragment is syntactically correct. Instead, it is *a runtime error*, and it produces a Traceback message that shows the context of the error, followed by the name of the error, IndexError, and a brief explanation.

Let's take a closer look at slicing, using our artificial sentence again. Here we verify that the slice 5:8 includes sent elements at indexes 5, 6, and 7:

sent[5:8]

By convention, m:n means elements m…n-1 inclusive. As the next example shows, we can omit the first number if the slice begins at the start of the list, and we can omit the second number if the slice goes to the end:

sent[:3]

text2[141525:]

We can modify an element of a list by assigning to one of its index values. In the next example, we put `sent[0]` on the left of the equals sign. We can also replace an entire slice with new elements. A consequence of this last change is that the list only has four elements, and accessing a later value generates an error.

sent[0] = 'First'
sent[9] = 'Last'
len(sent)

sent[1:9] = ['Second', 'Third']
sent

sent[9]

```{admonition} Your Turn
Take a few minutes to define a sentence of your own and modify individual words and groups of words (slices) using the same methods used earlier. Check your understanding by trying the exercises on lists at the end of this chapter.
```


2.3   Variables

The basic form of Python statements is: `variable = expression`. Python will evaluate the expression, and save its result to the variable. This process is called assignment. It does not generate any output; you have to type the variable on a line of its own to inspect its contents. The equals sign is slightly misleading, since information is moving from the right side to the left. It might help to think of it as a left-arrow. The name of the variable can be anything you like, e.g., my_sent, sentence, xyzzy. It must start with a letter, and can include numbers and underscores. Here are some examples of variables and assignments:

```{note}
Remember that capitalized words appear before lowercase words in sorted lists.
```

Notice Python expressions can be split across multiple lines, so long as this happens within any kind of brackets. It doesn't matter how much indentation is used in these continuation lines, but some indentation usually makes them easier to read.

It is good to choose meaningful variable names to remind you — and to help anyone else who reads your Python code — what your code is meant to do. Python does not try to make sense of the names; it blindly follows your instructions, and does not object if you do something confusing, such as `one = 'two'` or `two = 3`. The only restriction is that a variable name cannot be any of Python's reserved words, such as `def`, `if`, `not`, and `import`. If you use a reserved word, Python will produce a syntax error:

not = 'Camelot' 

We will often use variables to hold intermediate steps of a computation, especially when this makes the code easier to follow. Thus `len(set(text1))` could also be written:

vocab = set(text1)
vocab_size = len(vocab)
vocab_size

```{caution}
Take care with your choice of names (or identifiers) for Python variables. First, you should start the name with a letter, optionally followed by digits (0 to 9) or letters. Thus, abc23 is fine, but 23abc will cause a syntax error. Names are case-sensitive, which means that myVar and myvar are distinct variables. Variable names cannot contain whitespace, but you can separate words using an underscore, e.g., my_var. Be careful not to insert a hyphen instead of an underscore: my-var is wrong, since Python interprets the "-" as a minus sign.
```

## Strings
Some of the methods we used to access the elements of a list also work with individual words, or strings. For example, we can assign a string to a variable [1], index a string, and slice a string:

name = 'Monty'
name[0]

name[:4]

We can also perform multiplication and addition with strings:

name * 2

name + '!'

We can join the words of a list to make a single string, or split a string into a list, as follows:

' '.join(['Monty', 'Python'])

'Monty Python'.split()

We now have two important building blocks — lists and strings — and are ready to get back to some language analysis.