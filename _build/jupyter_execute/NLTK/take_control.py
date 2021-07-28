Back to Python: Making Decisions and Taking Control
===================================================

So far, our little programs have had some interesting qualities: the ability to work with language, and the potential to save human effort through automation. A key feature of programming is the ability of machines to make decisions on our behalf, executing instructions when certain conditions are met, or repeatedly looping through text data until some condition is satisfied. This feature is known as control, and is the focus of this section.

4.1   Conditionals
Python supports a wide range of operators, such as < and >=, for testing the relationship between values. The full set of these relational operators is shown in 4.1.

```{list-table} Numerical Comparison Operators
:header-rows: 1

* - Operator	
  - Relationship
* - <	
  - less than
* - <=	
  - less than or equal to
* - ==	
  - equal to (note this is two "=" signs, not one)
* - !=	
  - not equal to
* - &gt;	
  - greater than
* - &gt;=	
  - greater than or equal to
```


We can use these to select different words from a sentence of news text. Here are some examples — only the operator is changed from one line to the next. They all use `sent7`, the first sentence from text7 (Wall Street Journal). As before, if you get an error saying that `sent7` is undefined, you need to first type: `from nltk.book import *`

from nltk.book import *

sent7

[w for w in sent7 if len(w) < 4]

[w for w in sent7 if len(w) <= 4]

[w for w in sent7 if len(w) == 4]

[w for w in sent7 if len(w) != 4]

There is a common pattern to all of these examples: `[w for w in text if condition ]`, where condition is a Python "test" that yields either true or false. In the cases shown in the previous code example, the condition is always a numerical comparison. 

Instead of writing your own Regex, we can test various properties of words, using the functions for a string object listed below.

```{list-table} Some Word Comparison Operators
:header-rows: 1
* - Function	
  - Meaning
* - s.startswith(t)	
  - test if s starts with t
* - s.endswith(t)	
  - test if s ends with t
* - t in s	
  - test if t is a substring of s
* - s.islower()	
  - test if s contains cased characters and all are lowercase
* - s.isupper()	
  - test if s contains cased characters and all are uppercase
* - s.isalpha()	
  - test if s is non-empty and all characters in s are alphabetic
* - s.isalnum()	
  - test if s is non-empty and all characters in s are alphanumeric
* - s.isdigit()	
  - test if s is non-empty and all characters in s are digits
* - s.istitle()	
  - test if s contains cased characters and is titlecased (i.e. all words in s have initial capitals)
```
Here are some examples of these operators being used to select words from our texts: words ending with -`ableness`; words containing `gnt`; words having an initial capital; and words consisting entirely of digits.

sorted(w for w in set(text1) if w.endswith('ableness'))

sorted(term for term in set(text4) if 'gnt' in term)

sorted(item for item in set(text6) if item.istitle())[-5:]

sorted(item for item in set(sent7) if item.isdigit())

We can also create more complex conditions. If `c` is a condition, then `not c` is also a condition. If we have two conditions `c1` and `c2`, then we can combine them to form a new condition using conjunction and disjunction: `c1 and c2`, `c1 or c2`.

```{admonition} Your Turn
Run the following examples and try to explain what is going on in each one. Next, try to make up some conditions of your own.
```

```python
sorted(w for w in set(text7) if '-' in w and 'index' in w)
sorted(wd for wd in set(text3) if wd.istitle() and len(wd) > 10)
sorted(w for w in set(sent7) if not w.islower())
sorted(t for t in set(text2) if 'cie' in t or 'cei' in t)
```

## Operating on Every Element
We saw some examples of counting items other than words. Let's take a closer look at the notation we used:

```{python} 	
[len(w) for w in text1]
[w.upper() for w in text1]
```
```{note}
These expressions have the form [f(w) for ...] or [w.f() for ...], where f is a function that operates on a word to compute its length, or to convert it to uppercase. For now, you don't need to understand the difference between the notations f(w) and w.f(). Instead, simply learn this Python idiom which performs the same operation on every element of a list. In the preceding examples, it goes through each word in text1, assigning each one in turn to the variable w and performing the specified operation on the variable.

The notation just described is called a **list comprehension**." This is a Python idiom, a fixed notation that we use habitually without bothering to analyze each time. Mastering such idioms is an important part of becoming a fluent Python programmer.
```
Let's return to the question of vocabulary size, and apply the same idiom here:

len(text1)

len(set(text1))

len(set(word.lower() for word in text1))

Now that we are not double-counting words like This and this, which differ only in capitalization, we've wiped 2,000 off the vocabulary count! We can go a step further and eliminate numbers and punctuation from the vocabulary count by filtering out any non-alphabetic items:

len(set(word.lower() for word in text1 if word.isalpha()))

This example is slightly complicated: it lowercases all the purely alphabetic items. Perhaps it would have been simpler just to count the lowercase-only items, but this gives the wrong answer (why?).

## Nested Code Blocks
Most programming languages permit us to execute a block of code when a conditional expression, or if statement, is satisfied. We already saw examples of conditional tests in code like `[w for w in sent7 if len(w) < 4]`. In the following program, we have created a variable called word containing the string value 'cat'. The if statement checks whether the test `len(word) < 5` is true. It is, so the body of the if statement is invoked and the print statement is executed, displaying a message to the user. Remember to indent the print statement by typing four spaces.

word = 'cat'
if len(word) < 5:
    print('word length is less than 5')
else:
    print('word length is greater than or equal to 5')    

An if statement is known as a control structure because it controls whether the code in the indented block will be run. Another control structure is the `for` loop. Try the following, and remember to include the colon and the four spaces:

for word in ['Call', 'me', 'Ishmael', '.']:
   print(word)

This is called a loop because Python executes the code in circular fashion. It starts by performing the assignment word = 'Call', effectively using the word variable to name the first item of the list. Then, it displays the value of word to the user. Next, it goes back to the for statement, and performs the assignment word = 'me', before displaying this new value to the user, and so on. It continues in this fashion until every item of the list has been processed. 

```{tip}
If we want to create a list by iterating through a list, "list comprehension" is a Pythonic way and preferred than a loop.
```

## Looping with Conditions
Now we can combine the if and for statements. We will loop over every item of the list, and print the item only if it ends with the letter l. We'll pick another name for the variable to demonstrate that Python doesn't try to make sense of variable names.

sent1 = ['Call', 'me', 'Ishmael', '.']
for xyzzy in sent1:
    if xyzzy.endswith('l'):
        print(xyzzy)

You will notice that if and for statements have a colon at the end of the line, before the indentation begins. In fact, all Python control structures end with a colon. The colon indicates that the current statement relates to the indented block that follows.

We can also specify an action to be taken if the condition of the if statement is not met. Here we see the elif (else if) statement, and the else statement. Notice that these also have colons before the indented code.

for token in sent1:
    if token.islower():
        print(token, 'is a lowercase word')
    elif token.istitle():
        print(token, 'is a titlecase word')
    else:
        print(token, 'is punctuation')

As you can see, even with this small amount of Python knowledge, you can start to build multiline Python programs. It's important to develop such programs in pieces, testing that each piece does what you expect before combining them into a program. This is why the Python interactive interpreter is so invaluable, and why you should get comfortable using it.

Finally, let's combine the idioms we've been exploring. First, we create a list of cie and cei words, then we loop over each item and print it. Notice the extra information given in the print statement: end=' '. This tells Python to print a space (not the default newline) after each word.

tricky = sorted(w for w in set(text2) if 'cie' in w or 'cei' in w)
for word in tricky:
    print(word, end=' ')