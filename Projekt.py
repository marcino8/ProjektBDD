import threading
import pandas as pd
import numpy as np
import math
import sys
from graphviz import Digraph

node_counter = 0


# V2: LIMITATION
class Tree:
    """
        Summary or Description of the Class

        Attributes:
            Node root: node from which the tree starts

        Methods:
            print_diagram(self)
            gather_info(self, node, result)
            print_leafs(self)
            _print_leafs(self, node, result)
            compute_bdd(self)
            _compute_bdd(self, node)
            splitByEntropyTable(frame, E)
            calc_entr(frame)
            calc_main_entr(frame)

        Description:
            This is the class containing whole structure of
            Binary Decision Diagram.

        Required libraries:
            threading
            math
            pandas
            numpy
            sys
            graphviz
    """

    def __init__(self, node):
        self.root = node

    def print_diagram(self):
        """
            Summary or Description of the Function

            Description:
                Function creates graphic representation of BDD based on values gathered
                in gather_info func. This requires graphviz lib

        """
        info = []  # Zmienna na wyniki
        node = self.root
        self.gather_info(node, info)  # Zbierz informacje
        dot = Digraph(comment='BDD')  # Stwórz diagram
        dot.attr('node', shape='box', style='filled', color='lightgrey')
        dot.node("info", "LEGENDA \n X == y \n Jeśli tak: idź w prawo | Jeśli nie: idź w lewo")
        for v in info:
            if len(v) > 2:
                dot.node(v[0], v[1])
                dot.edge(v[0], v[2])
                dot.edge(v[0], v[3])
            else:
                dot.node(v[0], v[1])
        dot.render('test-output/diagram.gv', view=True)  # doctest: +SKIP
        'test-output/diagram.pdf'

    def gather_info(self, node, result):
        """
            Summary or Description of the Function

            Parameters:
                Node node: Current node in the recursion throughout tree structure
                List of Lists result: List to pass the information gathered

            Description:
                Function iterates throughout whole tree and for every node it gets:
                    Node id
                    What was dataframe in this node divided by
                    Left child id
                    Right child id
                All of it is put into a list, and in form of list put into result.
                If node has no children, then information gathered is:
                    Node id
                    Set of values in result column of the dataframe in value variable of the node
        """
        if node.has_next():  # Czy sa dalsze podzialy?
            #  Jeśli są dalsze podzialy, zbierz informacje o id, i o id polaczonych wezlow, oraz przez co
            #  dzielony byl dataframe w wezle
            result.append(
                [str(node.id), node.divided_by, str(node.left.id), str(node.right.id)])
            if node.has_right_child():
                self.gather_info(node.right, result)
            if node.has_left_child():
                self.gather_info(node.left, result)
        else:
            #  Jeśli nie ma dalszych podzialow, zbierz tylko informacje o id, i w tym wypadku divided_by
            #  To zbior wartosci kolumny wynikowej
            result.append([str(node.id), node.divided_by])

    def print_leafs(self):
        """
        Summary or Description of the Function

            Description:
                Function prints out leafes of a binary tree. If any node
                in the binary tree has no children it is considered a leaf

        """
        leafs = []  # Zmienna na wyniki
        node = self.root
        self._print_leafs(node, leafs)
        for leaf in leafs:
            print(leaf)
            print()

    def _print_leafs(self, node, result):
        """
        Summary or Description of the Function

            Description:
                Function includes recursion that helps func "print_leafs"

        """
        if node.has_next():  # Czy sa dalsze podzialy?
            if node.has_right_child():  # Jeśli tak to sprawdz to samo dla dzieci
                self._print_leafs(node.right, result)
            if node.has_left_child():
                self._print_leafs(node.left, result)
        else:  # Jeśli nie ma to dodaj dataframe do wynikow
            result.append(node.value)

    def compute_bdd(self):
        """
            Summary or Description of the Function

            Description:
                Function starts computing Binary Decision  based on
                Dataframe in self.root.value

        """
        self._compute_bdd(self.root)

    def _compute_bdd(self, node):
        """
            Summary or Description of the Function

            Parameters:
                Node node: Node of a tree

            Description:
                Function computes entropy tables for dataframe in node.value.
                Then it splits dataframe using splitByEntropyTable func. And puts
                the results as children of the current node(node).
                This happens untill splitByEntropyTable does not return 0s.

        """
        if len(set(node.value[node.value.columns[-1]])) != 1:
            # Jesli w dataframie w kolumnie wynikowej sa rozne wartosci
            # Kiedy beda takie same to nie liczymy entropii zeby zaoszczedzic pamieci
            co_dzielic = self.calc_entr(node.value)
            # Zapisz wg czego zostanie podzielony dataframe, w formie "Czy P == 1"
            wg_czego_dzielic = splitNameString(str(co_dzielic.idxmax(axis=1)[0]))
            node.divided_by = "Czy " + wg_czego_dzielic[0] + " == " + wg_czego_dzielic[1]
            podzielone = self.splitByEntropyTable(node.value, co_dzielic)  # Podziel dataframe wg maksymalnego I-Ej
            # Stworz nowe wezly i dolacz je do drzewa
            node_p = Node(podzielone[0])
            node_l = Node(podzielone[1])
            node.right = node_p
            node.left = node_l
            # Dla nowych wezlow obliczaj kolejne podzialy
            self._compute_bdd(node.right)
            self._compute_bdd(node.left)
        else:  # Jesli w dataframie w kolumnie wynikowej jest jedna wartosc, to skoncz pod rekurencje
            node.divided_by = "Wybierz: " + str(set(node.value[node.value.columns[-1]]).pop())

    @staticmethod
    def splitByEntropyTable(frame, E):
        """
            Summary or Description of the Function

            Parameters:`
                pandas.DataFrame frame: Dataframe to divide
                pandas.DataFrame E: Dataframe of calculated decision factors for frame

            Returns:
                List of 2 Dataframes, result of spliting parameter frame

            Description:
                Function splits Dataframe into 2 Dataframes by value in column
                which has biggest calculated decision factor.
                              E.g. If
                      W1      W2      P1      P2
                      0.4     0.2     0.4     0.7
                is E parameter, func splits parameter frame by column P
                and P value 2. So the results are 2 tables:
                    One with only 2s in P column
                    Second with every value but 2 in P column

        """
        podzielone = []  # Zmienna na result
        wg_czego_dzielic = splitNameString(str(E.idxmax(axis=1)[0]))  # Nazwa kolumny z maksymalnym wspolczynnikiem
        # I-Ej (Xn)

        value = int(wg_czego_dzielic[1])  # Wartosc w kolumnie wg ktorej dzielimy (n)
        wg_czego_dzielic = wg_czego_dzielic[0]  # Nazwa kolumny wg ktorwej dzielimy frame  (X)
        # Podzial
        podzielone.append(frame[frame[wg_czego_dzielic] == value])
        podzielone.append(frame[frame[wg_czego_dzielic] != value])
        return podzielone

    @staticmethod
    def calc_main_entr(frame):
        """
            Summary or Description of the Function

            Parameters:
            pandas.DataFrame frame: Dataframe to calculate main entropy

            Description:
            Function calculates main entropy of the given dataframe
        """
        entropy_of_df = 0
        # Obliczenie Entropii całego ukladu
        for value in set(frame[frame.columns[-1]]):
            n = len(frame[frame[frame.columns[-1]] == value])
            N = len(frame[frame.columns[-1]])
            entropy_of_df += (-n / N) * math.log(n / N, 2)
        return entropy_of_df

    @staticmethod
    def calc_entr(frame):
        """
            Summary or Description of the Function

            Parameters:
            pandas.DataFrame frame: Dataframe to calculate decision factors

            Description:
            Function calculates entropies for each value in each column without
            last column(where decision result is supposed to be given).
            E.g. if

            W       P       Result
            1       1       3
            1       2       2
            2       2       1

            is parameter frame, the result calculates factors decision for:
            value 1 in W, value 2 in W, value 1 in P and value 2 in P

            result dataframe will look like:

            W1      W2      P1      P2
            I-Ew1    I-Ew2    I-Ep1    I-Ep2

            where:
            I is entropy of the whole dataframe
            Exj is entropy of variable X of value j
        """
        entropy_of_df = Tree.calc_main_entr(frame)  # Wyliczenie entropii ogolnej

        # Wyliczanie E_j
        E = []
        kolumny = []
        for variable in frame.columns[0:-1]:  # Dla kazdej przesłanki znajdujacej sie w frame
            if len(set(frame[variable])) != 1:  # Jesli wartośc przesłanki nie jest taka sama w całej kolumnie
                for value in set(frame[variable]):  # Dla kazdej wartosci danej przeslanki
                    N = len(frame[frame[variable] == value])  # Ile jest danych w kolumnie przesłanka o wartosci value
                    sum_plus = 0
                    sum_minus = 0
                    for st in set(frame[frame.columns[-1]]):  # Dla kazdej wartosci kolumny wynikowej
                        pom = frame[frame[variable] == value]
                        pom2 = frame[frame[variable] != value]
                        # Ile jest danych w kolumnie przesłanka o wartosci value
                        # I o wartosci st w kolumnie wynikowej
                        n_plus = len(pom[pom[frame.columns[-1]] == st])
                        # Ile jest danych w kolumnie przesłanka o wartosci roznej od value
                        # I o wartosci st w kolumnie wynikowej
                        n_minus = len(pom2[pom2[frame.columns[-1]] == st])
                        if n_plus == 0:
                            Iplus = 0
                        else:
                            Iplus = (-n_plus / N) * math.log(n_plus / N, 2)  # Oblicz część I+
                        if n_minus == 0:
                            Iminus = 0
                        else:
                            Iminus = (-n_minus / (len(frame[frame.columns[-1]]) - N)) \
                                     * math.log(n_minus / (len(frame[frame.columns[-1]]) - N), 2)  # Oblicz część I-
                        sum_plus += Iplus  # Dodaj odpowiednio tak, ze po zakonczeniu tej petli w sumaplus
                        sum_minus += Iminus  # bedzie Ij+ a w sumaminus Ij-
                    Eip = (N / len(frame[frame.columns[-1]]))
                    Eim = (len(frame[frame.columns[-1]]) - N) / len(frame[frame.columns[-1]])
                    E.append(Eip * sum_plus + Eim * sum_minus)  # Oblicz Ej
                    kolumny.append(variable + "/" + str(value))  # Oznacz kolumne w dataframe np (W3)
        E = pd.DataFrame(E)
        E = E.transpose()
        E.columns = kolumny
        E = np.repeat(entropy_of_df, len(kolumny)) - E  # Wyniki I-Ej
        return E


class Node:
    """
        Summary or Description of the Class

        Attributes:
            Node left: value of left child in tree structure
            Node right: value of right child in tree structure
            pandas.DataFrame value: value in node
            int id: counter for the generated nodes in the program
            string divided_by: string formula determining by what dataframe in value was divided
                if Node has no children, variable contains set of values of result column in value dataframe

        Methods:
            has_next(self)
            has_right_child(self)
            has_left_child(self)

        Description:
            A node of a tree structure

        Limitation:
            *** No "/" marks in column names ***

    """

    def __init__(self, df):
        global node_counter
        node_counter = 1 + node_counter
        self.id = node_counter
        self.left = None
        self.right = None
        self.value = df
        self.divided_by = ""

    def has_next(self):
        """
            Summary or Description of the Function

            Returns:
            bool

            Description:
            Determines whether node has any children


        """
        return (self.right is not None) | (self.left is not None)

    def has_right_child(self):
        """
            Summary or Description of the Function

            Returns:
            bool

            Description:
            Determines whether node has right child

        """
        return self.right is not None

    def has_left_child(self):
        """
            Summary or Description of the Function

            Returns:
            bool

            Description:
            Determines whether node has left child

        """
        return self.left is not None


def splitNameString(name):
    """
        Summary or Description of the Function

        Parameters:
            :param name: string to divide

        Description:
            Divides string by '/'

    """
    return name.split(sep="/")


def sample_use(recursion_limit, file, rename_columns=False, rename_list=None, indexes_in_first_column=False):
    """
        Description:
            Function demonstrates how the program works
            :param indexes_in_first_column: bool decide whether first column of your dataframe
                contains rows indexing
            :param rename_list: list of strings decide the names of columns
            :param rename_columns: bool decide whether to rename columns or no
            :param file: string path to csv dataframe to read and to put into root.value of Tree
            :param recursion_limit: int(+) override system default recursion limit(10**4)
                if the input table is too big.

    """
    sys.setrecursionlimit(recursion_limit)
    df = pd.read_csv(file)
    if indexes_in_first_column:
        df = df.iloc[:, 1:7]
    if rename_columns:
        df.columns = rename_list
    drzewko = Tree(Node(df))
    threading.stack_size(200000000)
    thread = threading.Thread(target=drzewko.compute_bdd())
    thread.start()
    drzewko.print_diagram()
    drzewko.print_leafs()


sample_use(10 ** 6, "BDD.csv", True, ['P', 'W', 'B', 'O', 'PR', 'ST'], True)
