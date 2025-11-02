# Harjoitus 2 — Advanced Algorithm Challenge (Claude-4.5 Level)

## Tehtävän tarkoitus

Tämä on **erittäin haastava** versio alkuperäisestä harjoituksesta, suunniteltu testaamaan huipputason kielimallien kykyjä. Funktiot sisältävät edistyneitä algoritmeja lukuteoriasta, graafiteoriasta, merkkijonoalgoritmiteista, numeerisista menetelmistä ja optimoinnista.

## Sisältö

- `originals.py` — Oikeat, toimivat toteutukset (vertailuperusta)
- `buggy.py` — Hienostuneesti bugisia implementaatioita
- `grader.py` — Automaattinen arvioija
- `Tehtävänanto.md` — Tämä tiedosto

## Tehtävänanto

### 1. Luo ratkaisu

Luo tiedosto nimeltä `ratkaisu.py` samaan kansioon. Toteuta kaikki seuraavat funktiot:

#### Lukuteoria
- **`extended_gcd(a: int, b: int) -> Tuple[int, int, int]`**  
  Laajennettu Eukleideen algoritmi. Palauttaa `(gcd, x, y)` missä `gcd = ax + by`.
  
- **`chinese_remainder_theorem(remainders: List[int], moduli: List[int]) -> int`**  
  Ratkaisee kongruenssiyhtälöryhmän kiinalaisella jäännöslauseella.
  
- **`pollard_rho(n: int, max_iter: int = 100000) -> Optional[int]`**  
  Pollardin rho-algoritmi lukujen tekijöihinjakoon.
  
- **`miller_rabin(n: int, k: int = 5) -> bool`**  
  Miller-Rabin alkulukutesti probabilistisella menetelmällä.

#### Numeeriset algoritmit
- **`fast_inverse_sqrt(x: float) -> float`**  
  Quake III:n kuuluisa fast inverse square root -algoritmi.
  
- **`fft(signal: List[complex]) -> List[complex]`**  
  Cooley-Tukey FFT (Fast Fourier Transform).
  
- **`karatsuba_multiply(x: int, y: int) -> int`**  
  Karatsuban algoritmi nopeaan kertolaskuun.

#### Graafialgoritmit
- **`dijkstra_shortest_path(graph: Dict[int, List[Tuple[int, float]]], start: int, end: int) -> Tuple[float, List[int]]`**  
  Dijkstran algoritmi lyhimmän polun löytämiseen + polun rekonstruktio.
  
- **`convex_hull(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]`**  
  Graham scan -algoritmi konveksin käyrän laskemiseen.

#### Merkkijonoalgoritmit
- **`knuth_morris_pratt(text: str, pattern: str) -> List[int]`**  
  KMP-algoritmi merkkijonojen etsintään.
  
- **`suffix_array(s: str) -> List[int]`**  
  Rakenna suffiksitaulukko prefix doubling -menetelmällä.
  
- **`aho_corasick_search(text: str, patterns: List[str]) -> Dict[str, List[int]]`**  
  Aho-Corasick -algoritmi usean merkkijonon etsintään.

#### Dynaaminen ohjelmointi
- **`longest_increasing_subsequence(arr: List[int]) -> int`**  
  Pisimmän kasvavan alijonon pituus (binäärihaku-optimointi).

#### Matriisioperaatiot
- **`matrix_determinant(matrix: List[List[float]]) -> float`**  
  Matriisin determinantti kofaktorikehitelmällä.

#### Optimointi
- **`simplex_method(c: List[float], A: List[List[float]], b: List[float]) -> Optional[Tuple[float, List[float]]]`**  
  Simpleksi-algoritmi lineaariseen ohjelmointiin.

### 2. Tallenna ja testaa

```bash
python grader.py
```

Grader suorittaa kattavan testisarjan jokaiselle funktiolle, mukaan lukien:
- Edge caset
- Numeeriset tarkkuushaasteet
- Algoritminen oikeellisuus
- Suorituskyky suurilla syötteillä

## Haasteet

### 🔥 Miksi tämä on vaikeaa?

1. **Hienostuneita bugeja**: Bugit eivät ole ilmeisiä. Ne sisältävät:
   - Väärät etumerkit kriittisissä kohdissa
   - Off-by-one -virheet monimutkaisissa silmukoissa
   - Virheelliset taikavakiot
   - Puuttuvat reunaehdot
   - Väärin toteutetut matemaattiset kaavat

2. **Algoritminen syvyys**: Vaatii ymmärrystä:
   - Modulaariaritmetiikasta
   - Fourier-analyyseista
   - Graafiteoriasta
   - Laskennallisesta geometriasta
   - Merkkijonojen pattern matching -teoriasta
   - Optimointiteoriasta

3. **Numeerinen tarkkuus**: Useat algoritmit vaativat:
   - Floating-point -aritmetiikan hallintaa
   - Kompleksilukujen käsittelyä
   - Tarkkuuden säilyttämistä iteratiivisissa algoritmeissa

4. **Ohjelmointitaidot**: Vaatii hallintaa:
   - Rekursiosta ja divide-and-conquer -strategioista
   - Dynaamisesta ohjelmoinnista
   - Bittimanipulaatiosta
   - Tietorakenteiden toteuttamisesta (tries, heaps)

## Rajoitteet ja vaatimukset

- **Python 3.12+**
- **Vain standardikirjasto** (math, cmath, collections, heapq, struct, random, ast)
- **Funktiosignatuurit** täytyy säilyttää täsmälleen
- **Tyypitysmerkinnät** suositeltavia
- **Suorituskyky** huomioitava (jotkut testit käyttävät suuria syötteitä)

## Arviointi

- **15 funktiota** yhteensä
- **5-7 testiä** per funktio
- Testi hyväksytään vain jos **kaikki** sen testit menevät läpi
- **Toleranssi**: ±1e-5 liukuluvuille ja kompleksiluvuille

### Arvosana-asteikko
- **100%**: 🎉 Täydellinen! Huipputason suoritus!
- **80-99%**: 🌟 Erinomainen! Lähes kaikki hallussa!
- **60-79%**: 👍 Hyvä! Jatka harjoittelua!
- **<60%**: 💪 Haasta itsesi enemmän!

## Vinkit onnistumiseen

### Debuggaus-strategia
1. **Ymmärrä algoritmi**: Lue teoria ensin (esim. Wikipedia, CLRS, algorithmic resources)
2. **Vertaa implementaatioita**: Katso sekä `buggy.py` että `originals.py`
3. **Eristä bug**: Käytä print-debuggausta tai debuggeria
4. **Testaa pienillä syötteillä**: Käy läpi algoritmi käsin
5. **Tarkista reunaehdot**: Tyhjät syötteet, yhden elementin syötteet, jne.

### Algoritmit joissa yleensä virheitä
- **Extended GCD**: Etumerkkien käsittely negatiivisilla luvuilla
- **Matrix Determinant**: Alternoivat etumerkit kofaktorikehitelmässä
- **Fast Inverse Sqrt**: Oikean taikavakion käyttö
- **CRT**: Modulaarinen käänteisluvun laskenta
- **Pollard Rho**: Oikean pseudosatunnaisgeneraattorin käyttö
- **FFT**: Twiddle-faktorin etumerkki
- **Karatsuba**: Rekombinointikaava
- **Dijkstra**: Polun päivitys lyhyemmän reitin löytyessä
- **Convex Hull**: Cross product -laskenta
- **LIS**: Binary search boundary conditions
- **KMP**: LPS-taulukon rakentaminen
- **Simplex**: Unboundedness-tarkistus
- **Miller-Rabin**: Witness-tarkistuksen logiikka
- **Aho-Corasick**: Failure link -rakentaminen

### Jos käytät LLM:ää
1. Anna **täsmälliset speksit**: Funktioiden allekirjoitukset, odotettu toiminta
2. **Pyydä selitys**: Älä vain kopioi koodia, ymmärrä miksi se toimii
3. **Testaa iteratiivisesti**: Korjaa yksi funktio kerrallaan
4. **Vertaa algoritmeja**: Pyydä LLM:ää selittämään ero bugisen ja oikean version välillä

## Esimerkkejä

### Extended GCD
```python
# Oikein:
extended_gcd(48, 18) → (6, -1, 3)  # 6 = 48*(-1) + 18*3

# Buginen versio saattaa antaa väärät kertoimet
```

### FFT
```python
# Syöte: [1+0j, 1+0j, 1+0j, 1+0j]
# Oikea tulos: [4+0j, 0+0j, 0+0j, 0+0j]
```

### Dijkstra
```python
graph = {
    0: [(1, 4), (2, 1)],
    1: [(3, 1)],
    2: [(1, 2), (3, 5)],
    3: []
}
dijkstra_shortest_path(graph, 0, 3)
# → (4.0, [0, 2, 1, 3])  # Lyhin polku ja sen pituus
```

## Lisäresurssit

- **CLRS**: Introduction to Algorithms (Cormen et al.)
- **Wikipedia**: Lähes kaikille algoritmeille on hyvät artikkelit
- **GeeksforGeeks**: Käytännön implementaatioesimerkkejä
- **CP-Algorithms**: Kilpaohjelmointialgoritmeja
- **Project Euler**: Matemaattisia haastelaskuja

## Hauskaa koodaamista! 🚀

Muista: Nämä ovat algoritmeja, joita käytetään oikeissa tuotantojärjestelmissä (kryptografia, signaalinkäsittely, tietokannat, kompressointi, jne.). Niiden hallitseminen tekee sinusta paremman ohjelmoijan!
