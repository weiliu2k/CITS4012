Python Basics
=============

https://www.w3schools.com/python/

# input and output
name = input()
print("hello, " + name)

# print multiple variables separated by a space
print("hello", name, 1, 3.0, True)

# line comment
"""
block 
comments
"""

# variables don't need explicit declaration
var = "hello" # string
var = 10.0    # float
var = 10      # int
var = True    # boolean
var = [1,2,3] # pointer to list
var = None    # empty pointer

# type conversions
var = 10
print(int(var))
print(str(var))
print(float(var))

# basic math operations
var = 10
print("var + 4 =", 10 + 4)
print("var - 4 =", 10 - 4)
print("var * 4 =", 10 * 4)
print("var ^ 4=", 10 ** 4)
print("int(var) / 4 =", 10//4)   # // for int division
print("float(var) / 4 =", 10/4)  # / for float division
# All compound assignment operators available
# including += -= *= **= /= //= 
# pre/post in/decrementers not available (++ --)

# basic boolean operations include "and", "or", "not"
print("not True is", not True)
print("True and False is", True and False)
print("True or False is", True or False)

# String operations
# '' and "" are equivalent
s = "String"
#s = 'Mary said "Hello" to John'
#s = "Mary said \"Hello\" to John"

# basic
print(len(s)) # get length of string and any iterable type
print(s[0]) # get char by index
print(s[1:3]) # [1,3)
print("This is a " + s + "!")

# handy tools
print(s.lower())
print(s*4)
print("ring" in s)
print(s.index("ring"))

# slice by delimiter
print("I am a sentence".split(" "))
# concatenate a list of string using a delimiter
print("...".join(['a','b','c']))

# formatting variables
print("Formatting a string like %.2f"%(0.12345)) 
print(f"Or like {s}!")

# control flows
# NOTE: No parentheses or curly braces
#       Indentation is used to identify code blocks
#       So never ever mix spaces with tabs
for i in range(0,5):
    for j in range(i, 5):
        print("inner loop")
    print("outer loop")

# if-else
var = 10
if var > 10:
    print(">")
elif var == 10:
    print("=")
else:
    print("<")

# use "if" to check null pointer or empty arrays
var = None
if var: 
    print(var)
var = []
if var:
    print(var)
var = "object"
if var:
    print(var)

# while-loop
var = 5
while var > 0:
    print(var)
    var -=1

# for-loop
for i in range(3):  # prints 0 1 2
    print(i)
    
"""
equivalent to
for (int i = 0; i < 3; i++)
"""
print("-------")
# range (start-inclusive, stop-exclusive, step)
for i in range(2, -3, -2): 
    print(i)
"""
equivalent to
for (int i = 2; i > -3; i-=2)
"""

# define function
def func(a, b):
    return a + b
func(1,3)

# use default parameters and pass values by parameter name
def rangeCheck(a, min_val = 0, max_val=10):
    return min_val < a < max_val    # syntactic sugar
rangeCheck(5, max_val=4)

# define class
class Foo:
    
    # optinal constructor
    def __init__(self, x):
        # first parameter "self" for instance reference, like "this" in JAVA
        self.x = x
    
    # instance method
    def printX(self): # instance reference is required for all function parameters
        print(self.x)
        
    # class methods, most likely you will never need this
    @classmethod
    def printHello(self):
        print("hello")
        
obj = Foo(6)
obj.printX()

# class inheritance - inherits variables and methods
# You might need this when you learn more PyTorch
class Bar(Foo):
    pass
obj = Bar(3)
obj.printX()