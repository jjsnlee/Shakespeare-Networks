
# Shakespeare Networks

This is an old project from 2017.

- Website: https://skpn-jjsnlee.pythonanywhere.com/shakespeare/

- Paper: http://digitalhumanities.org:8081/dhq/vol/11/2/000289/000289.html

For every Scene of each Shakespeare's plays, this displays the relationships of the speakers as a graph using the Python library [networkx](https://networkx.org/). 

Nodes are the characters, and edges weighted by how often one character speaks after another. This is used as a rough proxy for characters speaking to one another; though in fact, in many cases characters do not speak to one another but in a group, or reacting as an aside. However the graphs still gives a decent sense of relationships between characters, and also the importance (or dominance) of certain characters in the plays.

## Example

###  Hamlet's monologue, after which he comes upon Ophelia alone -- others eavesdropping to assess his state of mind.

![.](https://skpn-jjsnlee.pythonanywhere.com/imgs/hamlet_3_1.png)

- Hamlet has the most lines in the scene.
- Hamlet and Ophelia are more proximate physically but also as dialogue partners.
- Polonius, Claudius, and others are proximate to one another, and distant from Hamlet and Ophelia. They are hidden behind Ophelia.
- Though Polonius does not interact directly with Hamlet, as he listens in he reacts to Hamlet's words most directly among the eavesdroppers. 


## Running it

If you would like to run this locally or enhance this for other plays you can:

```
> poetry install
> poety shell
> ./run-server.sh
```
