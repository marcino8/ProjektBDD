import threading
import pandas as pd
import numpy as np
import math
import sys
from graphviz import Digraph

node_counter = 0


class Tree:
    """
        Summary or Description of the Class

        Variables:
        Node root: node from which the tree starts

        Description:
        This is the class containing whole structure of
        Binary Decision Diagram

    """
    def __init__(self, node):
        self.root = node

    def zbierz_infromacje(self):
        info = []  # Zmienna na wyniki
        node = self.root
        self._zbierz_infromacje(node, info)
        return info

    def _zbierz_infromacje(self, wezel, wynik):
        if wezel.has_next():  # Czy sa dalsze podzialy?
            wynik.append(
                [str(wezel.id), wezel.podzielone, str(wezel.lewy.id), str(wezel.prawy.id)])
            if wezel.has_prawy():  # Jeśli tak to sprawdz to samo dla dzieci
                self._zbierz_infromacje(wezel.prawy, wynik)
            if wezel.has_lewy():
                self._zbierz_infromacje(wezel.lewy, wynik)
        else:
            wynik.append([str(wezel.id), wezel.podzielone])

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

    def _print_leafs(self, wezel, wynik):
        """
        Summary or Description of the Function

            Description:
            Function includes recursion that helps func "print_leafs"

        """
        if wezel.has_next():  # Czy sa dalsze podzialy?
            if wezel.has_prawy():  # Jeśli tak to sprawdz to samo dla dzieci
                self._print_leafs(wezel.prawy, wynik)
            if wezel.has_lewy():
                self._print_leafs(wezel.lewy, wynik)
        else:  # Jeśli nie ma to dodaj dataframe do wynikow
            wynik.append(wezel.wartosc)

    # TODO: PEWNIE SPLIT LEAFS FUNC

    def compute_bdd(self):
        """
            Summary or Description of the Function

            Description:
            Function starts computing Binary Decision  based on
            Dataframe in self.root.wartosc

        """
        self._compute_bdd(self.root)

    def _compute_bdd(self, wezel):
        """
            Summary or Description of the Function

            Parameters:
            Node wezel: Node of a tree

            Description:
            Function computes entropy tables for dataframe in wezel.wartosc.
            Then it splits dataframe using splitByEntropyTable func. And puts
            the results as children of the current node(wezel).
            This happens untill splitByEntropyTable does not return 0s.

        """
        if len(set(wezel.wartosc[wezel.wartosc.columns[-1]])) != 1:
            # Jesli w dataframie w kolumnie wynikowej sa rozne wartosci
            # Kiedy beda takie same to nie liczymy entropii zeby zaoszczedzic pamieci
            co_dzielic = self.calc_entr(wezel.wartosc)
            if sum(co_dzielic.iloc[0]) == 0: # Warunek koncowy, jesli wszystkie wspolczynniki decyzyjne
                wezel.podzielone = "WYNIK:"+str(set(wezel.wartosc[wezel.wartosc.columns[-1]]))
                return  # I-Ej sa rowne 0 to znaczy ze juz dalej nie dzielimy, wiec przerwij
            wg_czego_dzielic = str(co_dzielic.idxmax(axis=1)[0])
            wezel.podzielone = "Czy "+str(wg_czego_dzielic[:-1]) +"=="+str(int(wg_czego_dzielic[-1]))
            podzielone = self.splitByEntropyTable(wezel.wartosc, co_dzielic)  # Podziel dataframe wg maksymalnego I-Ej
            # Stworz nowe wezly i dolacz je do drzewa
            node_p = Node(podzielone[0])
            node_l = Node(podzielone[1])
            wezel.prawy = node_p
            wezel.lewy = node_l
            # Dla nowych wezlow obliczaj kolejne podzialy
            self._compute_bdd(wezel.prawy)
            self._compute_bdd(wezel.lewy)
        else:
            wezel.podzielone = "WYNIK:"+str(set(wezel.wartosc[wezel.wartosc.columns[-1]]))

    def splitByEntropyTable(self, frame, E):
        """
            Summary or Description of the Function

            Parameters:
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
        podzielone = []  # Zmienna na wynik
        wg_czego_dzielic = str(E.idxmax(axis=1)[0])  # Nazwa kolumny z maksymalnym wspolczynnikiem I-Ej (Xn)
        value = int(wg_czego_dzielic[-1])  # Nazwa kolumny wg ktorwej dzielimy frame  (X)
        wg_czego_dzielic = wg_czego_dzielic[:-1]  # Wartosc w kolumnie wg ktorej dzielimy (n)
        # Podzial
        podzielone.append(frame[frame[wg_czego_dzielic] == value])
        podzielone.append(frame[frame[wg_czego_dzielic] != value])
        return podzielone

    def calc_entr(self, frame):
        """
            Summary or Description of the Function

            Parameters:
            pandas.DataFrame frame: Dataframe to calculate decision factors

            Returns:
            pandas.DataFrame of calculated decision factors for frame Dataframe

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
            x1      x2      x3      x4


        """
        entropia_ukladu = 0
        # Obliczenie Entropii całego ukladu
        for value in set(frame[frame.columns[-1]]):
            n = len(frame[frame[frame.columns[-1]] == value])
            N = len(frame[frame.columns[-1]])
            entropia_ukladu += (-n / N) * math.log(n / N, 2)

        # Wyliczanie E_j
        E = []
        kolumny = []
        for przeslanka in frame.columns[0:-1]:  # Dla kazdej przesłanki znajdujacej sie w frame
            for value in set(frame[przeslanka]):  # Dla kazdej wartosci danej przeslanki
                sumaplus = 0
                sumaminus = 0
                N = len(frame[frame[przeslanka] == value])  # Ile jest danych w kolumnie przesłanka o wartosci value
                if len(set(frame[przeslanka])) != 1:
                    for st in set(frame[frame.columns[-1]]):  # Dla kazdej wartosci kolumny wynikowej
                        pom = frame[frame[przeslanka] == value]
                        pom2 = frame[frame[przeslanka] != value]
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
                        sumaplus += Iplus  # Dodaj odpowiednio tak, ze po zakonczeniu tej petli w sumaplus
                        sumaminus += Iminus  # bedzie Ij+ a w sumaminus Ij-
                Eip = (N / len(frame[frame.columns[-1]]))
                Eim = (len(frame[frame.columns[-1]]) - N) / len(frame[frame.columns[-1]])
                E.append(Eip * sumaplus + Eim * sumaminus)  # Oblicz Ej
                kolumny.append(przeslanka + str(value))  # Oznacz kolumne w dataframe np (W3)
        E = pd.DataFrame(E)
        E = E.transpose()
        E.columns = kolumny
        E = abs(np.repeat(entropia_ukladu, len(kolumny)) - E)  # Wyniki I-Ej
        E = E.replace(entropia_ukladu, 0)  # Jesli jakies Ej jest rowne entropii ukladu, to znaczy ze Ej = 0
        return E


class Node:
    """
        Summary or Description of the Class

        Variables:
        Node lewy: value of left child in tree structure
        Node prawy: value of right child in tree structure
        pandas.DataFrame wartosc: value in node

        Description:
        Function demonstrates how the program works

    """


    def __init__(self, df):
        global node_counter
        node_counter = 1 + node_counter
        self.id = node_counter
        self.lewy = None
        self.prawy = None
        self.wartosc = df
        self.podzielone = ""

    def has_next(self):
        """
            Summary or Description of the Function

            Returns:
            bool

            Description:
            Determines whether node has any children

        """
        return (self.prawy is not None) | (self.lewy is not None)

    def has_prawy(self):
        """
            Summary or Description of the Function

            Returns:
            bool

            Description:
            Determines whether node has right child

        """
        return self.prawy is not None

    def has_lewy(self):
        """
            Summary or Description of the Function

            Returns:
            bool

            Description:
            Determines whether node has left child

        """
        return self.lewy is not None


def sample_use(recursion_limit, file, rename_columns=False, rename_list=None):
    """
        Summary or Description of the Function

        Parameters:
        int(+) recursion_limit: override system default recursion limit(10**4)
            if the input table is too big.
        string file: path to csv dataframe to read and to put into root.wartosc of Tree
        optional bool rename_columns: decide whether to rename columns or no
        optional list of strings rename_list: decide the names of columns

        Description:
        Function demonstrates how the program works

    """
    sys.setrecursionlimit(recursion_limit)
    df = pd.read_csv(file)
    df = df.iloc[:, 1:7]
    if rename_columns:
        df.columns = rename_list
    drzewko = Tree(Node(df))
    threading.stack_size(200000000)
    thread = threading.Thread(target=drzewko.compute_bdd())
    thread.start()
    # drzewko.print_leafs()
    dot = Digraph(comment='The Round Table')
    dot.attr('node', shape='box')
    for v in drzewko.zbierz_infromacje():
            if len(v) > 2:
                dot.node(v[0], v[1])
                dot.edge(v[0], v[2])
                dot.edge(v[0], v[3])
            else:
                dot.node(v[0], v[1])
    dot.render('test-output/round-table.gv', view=True)  # doctest: +SKIP
    'test-output/round-table.gv.pdf'



sample_use(10 ** 6, "BDD.csv", True, ['P', 'W', 'B', 'O', 'PR', 'ST'])

#  TODO: ZAPYTAC O OSTATNIE PODZIALY BO ENTROPIE WYCHODZA 0 A DA SIE DZIELIC DALEJ
# framka = pd.DataFrame([[1, 1, 1, 1, 1], [1, 1, 2, 1, 1]])
# framka.columns = ["mama", "tata", "wujek", "brat", "siostra"]
# print(framka.to_string(index=False))
#
# dot = Digraph(comment='The Round Table')
# dot.attr('node', shape='box')
# dot.node('1A', framka.to_string(index=False))
# dot.node('2A', 'Sir Bedevere the Wise')
# dot.node('L', 'Sir Lancelot the Brave')
# dot.node('F', 'Cos tu mam')
# dot.edges(['LF', 'LF'])
# dot.edge('1A', '2A')
# print(dot.source)
#
# dot.render('test-output/round-table.gv', view=True)  # doctest: +SKIP
# 'test-output/round-table.gv.pdf'
#
# tree = Digraph(comment='Drzewo')
