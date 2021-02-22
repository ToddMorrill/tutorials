main :- write('hello, world!').

% rules (with recursion)
travel(A, C) :- next_to(A, B), next_to(B, C).
travel(A, B) :- next_to(A, B).
travel(A, B) :- next_to(A, X), travel(X, B).

% relations
next_to(california, nevada).
next_to(california, oregon).
next_to(california, arizona).
next_to(oregon, california).
next_to(nevada, california).
next_to(arizona, california).

% properties
fruit(orange).
fruit(apple).
fruit(pear).

