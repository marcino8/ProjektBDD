import threading
import pandas as pd
import numpy as np
import math
import sys


class Tree:
    def __init__(self, node):
        self.root = node

    def wypisz_liscie(self):
        liscie = []
        node = self.root
        self._wypisz_liscie(node, liscie)
        for leaf in liscie:
            print(leaf)
            print()

    def _wypisz_liscie(self, wezel, wynik):
        if wezel.has_next():
            if wezel.has_prawy():
                self._wypisz_liscie(wezel.prawy, wynik)
            if wezel.has_lewy():
                self._wypisz_liscie(wezel.lewy, wynik)
        else:
            wynik.append(wezel.wartosc)
    # TODO: PEWNIE SPLIT LEAFS FUNC
    def licz_drzewo_start(self):
        self._licz_drzewo(self.root)

    def _licz_drzewo(self, wezel):
        if len(set(wezel.wartosc[wezel.wartosc.columns[-1]])) != 1:
            co_dzielic = self.calc_entr(wezel.wartosc)
            if sum(co_dzielic.iloc[0]) == 0:  # JESLI WSZYSTKIE ENTROPIE SA ROWNE 0
                return
            podzielone = self.splitByEntropyTable(wezel.wartosc, co_dzielic)
            node_p = Node(podzielone[0])
            node_l = Node(podzielone[1])
            wezel.prawy = node_p
            wezel.lewy = node_l
            self._licz_drzewo(wezel.prawy)
            self._licz_drzewo(wezel.lewy)

    def splitByEntropyTable(self, frame, E):
        podzielone = []
        wg_czego_dzielic = str(E.idxmax(axis=1)[0])  # Wartosc wg ktorwej dzielimy
        value = int(wg_czego_dzielic[-1])
        wg_czego_dzielic = wg_czego_dzielic[:-1]
        podzielone.append(frame[frame[wg_czego_dzielic] == value])
        podzielone.append(frame[frame[wg_czego_dzielic] != value])
        return podzielone

    def calc_entr(self, frame):  # Przy załozeniu ze kolumna wynikow dzielonego df ma nazwe ST
        entropia_ukladu = 0
        # Entropia układu I
        for value in set(frame[frame.columns[-1]]):
            n = len(frame[frame[frame.columns[-1]] == value])
            N = len(frame[frame.columns[-1]])
            entropia_ukladu += (-n / N) * math.log(n / N, 2)

        # Wyliczanie E_j
        E = []
        kolumny = []
        for przeslanka in frame.columns[0:-1]:  # Bez kolumny wynikowej ST
            for value in set(frame[przeslanka]):
                sumaplus = 0
                sumaminus = 0
                N = len(frame[frame[przeslanka] == value])
                if len(set(frame[przeslanka])) != 1:
                    for st in set(frame[frame.columns[-1]]):
                        pom = frame[frame[przeslanka] == value]
                        pom2 = frame[frame[przeslanka] != value]
                        n_plus = len(pom[pom[frame.columns[-1]] == st])
                        n_minus = len(pom2[pom2[frame.columns[-1]] == st])
                        if n_plus == 0:
                            Iplus = 0
                        else:
                            Iplus = (-n_plus / N) * math.log(n_plus / N, 2)
                        if n_minus == 0:
                            Iminus = 0
                        else:
                            Iminus = (-n_minus / (len(frame[frame.columns[-1]]) - N)) \
                                     * math.log(n_minus / (len(frame[frame.columns[-1]]) - N), 2)
                        sumaplus += Iplus
                        sumaminus += Iminus
                Eip = (N / len(frame[frame.columns[-1]]))
                Eim = (len(frame[frame.columns[-1]]) - N) / len(frame[frame.columns[-1]])
                E.append(Eip * sumaplus + Eim * sumaminus)
                kolumny.append(przeslanka + str(value))
        E = pd.DataFrame(E)
        E = E.transpose()
        E.columns = kolumny
        E = abs(np.repeat(entropia_ukladu, len(kolumny)) - E)  # Wyniki I-Ej
        E = E.replace(entropia_ukladu, 0)
        return E


class Node:
    def __init__(self, df):
        self.rodzic = None
        self.lewy = None
        self.prawy = None
        self.wartosc = df

    def has_next(self):
        return (self.prawy is not None) | (self.lewy is not None)

    def has_prawy(self):
        return self.prawy is not None

    def has_lewy(self):
        return self.lewy is not None


def sample_use(recursion_limit, file, rename_columns=False, rename_list=None):
    sys.setrecursionlimit(recursion_limit)
    df = pd.read_csv(file)
    df = df.iloc[:, 1:7]
    if rename_columns:
        df.columns = rename_list
    drzewko = Tree(Node(df))
    threading.stack_size(200000000)
    thread = threading.Thread(target=drzewko.licz_drzewo_start())
    thread.start()
    drzewko.wypisz_liscie()


sample_use(10 ** 6, "BDD.csv", True, ['P', 'W', 'B', 'O', 'PR', 'ST'])

#  TODO: ZAPYTAC O OSTATNIE PODZIALY BO ENTROPIE WYCHODZA 0 A DA SIE DZIELIC DALEJ
