# Harjoitus 3 — Advanced Algorithm Optimization with AI

## Tavoite
Tämä on edistynyt versio optimointiharjoituksesta. Tehtävänäsi on optimoida monimutkaisia algoritmeja, jotka kattavat graafialgoritmeja, dynaamista ohjelmointia, merkkijonojen käsittelyä ja muita klassisia tietojenkäsittelytieteen ongelmia. Käytä Python 3.12:ta ja vain standardikirjastoa.

## Algoritmiset alueet

Tämä harjoitus sisältää funktioita seuraavista kategorioista:

### 1. Graafialgoritmit
- **dijkstra_shortest_path**: Lyhimmän polun etsintä painotetussa verkossa
- **topological_sort**: Topologinen järjestäminen suunnatussa asyklisessä verkossa
- **strongly_connected_components**: Vahvasti yhdistettyjen komponenttien etsintä
- **maximal_matching**: Maksimaalinen täsmäytys verkossa

### 2. Dynaaminen ohjelmointi
- **knapsack_01**: 0/1 Reppu-ongelma
- **longest_increasing_subsequence**: Pisin kasvava alijono
- **edit_distance**: Levenshtein-etäisyys kahden merkkijonon välillä
- **traveling_salesman_dp**: Kauppamatkustajan ongelma (TSP)
- **matrix_chain_multiplication**: Matriisiketjun kertolaskun optimointi

### 3. Merkkijonojen käsittely
- **suffix_array**: Suffiksitaulukon rakentaminen
- **rabin_karp_search**: Rabin-Karp merkkijonon etsintä
- **edit_distance**: Editointietäisyys operaatiotiedoilla

### 4. Tietorakenteet
- **lru_cache_simulator**: LRU-välimuistin simulointi
- **segment_tree_range_sum**: Segmenttipuu välisummakyselyille
- **bloom_filter_operations**: Bloom-suodattimen simulointi

### 5. Geometria ja matematiikka
- **convex_hull**: Konveksihullon laskenta
- **fast_fourier_transform**: Nopea Fourier-muunnos
- **prime_factorization**: Alkulukutekijöihin jako
- **interval_scheduling**: Aikavälisuunnitteluongelma

## Mitä paketissa on

- `originals.py` — Optimoidut viiteversiot (oikeat tulokset, tehokkaat algoritmit)
- `slow_funcs.py` — Tarkoituksella hitaat versiot samoille funktioille
- `ratkaisu_template.py` — Käynnistuspohja optimoiduille ratkaisuillesi
- `speed_compare.py` — Vertailuskripti suorituskyvyn mittaamiseen
- `Tehtävänanto.md` — Tämä tiedosto

## Tehtäväsi

1. **Tutki koodia**: Lue `slow_funcs.py` ja `originals.py` huolellisesti
2. **Tunnista pullonkaulat**: Etsi algoritmisista tehottomuuksista ja antipatternista
3. **Optimoi**: Luo `ratkaisu.py` ja toteuta optimoidut versiot
4. **Vertaile**: Käytä `speed_compare.py` mittaamaan parannuksia

## Optimointistrategiat

### Aikavaativuuden parantaminen
- O(n²) → O(n log n) käyttäen tehokkaita järjestämisalgoritmeja
- O(2ⁿ) → O(n²) tai O(n³) käyttäen dynaamista ohjelmointia
- O(n) → O(log n) käyttäen binäärihakua

### Yleisiä optimointitekniikoita
- **Muistiointi (Memoization)**: Tallenna lasketut arvot
- **Prioriteettijonot**: Käytä heapq-moduulia tehokkaaseen järjestyksenpitoon
- **Binäärihaku**: Hyödynnä järjestettyjä rakenteita
- **Tietorakenteiden valinta**: Käytä oikeita tietorakenteita (set, dict, deque)
- **Välttämättömien operaatioiden minimointi**: Älä toista samoja laskelmia

### Esimerkkejä

#### Ennen (O(n²))
```python
def find_duplicates(arr):
    result = []
    for i in range(len(arr)):
        for j in range(i+1, len(arr)):
            if arr[i] == arr[j]:
                result.append(arr[i])
    return result
```

#### Jälkeen (O(n))
```python
def find_duplicates(arr):
    seen = set()
    result = []
    for x in arr:
        if x in seen and x not in result:
            result.append(x)
        seen.add(x)
    return result
```

## Arviointi ja testaus

### Suorita testit
```pwsh
python speed_compare.py
```

### Mitä odotetaan
- **Nopeussuhde > 10x**: Erinomainen optimointi
- **Nopeussuhde 5-10x**: Hyvä optimointi
- **Nopeussuhde 2-5x**: Kohtalainen parannus
- **Nopeussuhde < 2x**: Tarvitsee lisäoptimointia

### Huomioitavaa
- Jotkin funktiot voivat olla helpompia optimoida kuin toiset
- Keskity algoritmisiin parannuksiin, ei mikrooptimointeihin
- Varmista, että tulokset ovat edelleen oikein

## Vinkkejä

### Graafialgoritmit
- Käytä `heapq` Dijkstran algoritmissa prioriteettijonona
- Tarjanin algoritmi on tehokkaampi kuin Kosarajun SCC:lle
- Kahnin algoritmi on intuitiivinen topologiseen järjestämiseen

### Dynaaminen ohjelmointi
- Rakenna taulukko alhaalta ylös rekursion sijaan (paitsi muistioinnilla)
- Optimoi tilankäyttöä käyttämällä 1D-taulukoita 2D:n sijaan
- Käytä tuple-avaimia muistiointiin moniulotteisissa ongelmissa

### Merkkijonot
- Rabin-Karp: Käytä vierivää hajautusta (rolling hash)
- Suffiksitaulukot: Harkitse tehokkaampaa rakennusalgoritmia
- Edit distance: Taulukoi iteratiivisesti

### Tietorakenteet
- Segmenttipuu: Rakenna puurakenne etukäteen
- Bloom filter: Minimoi hajautusfunktion uudelleenlaskenta
- LRU cache: Käytä OrderedDict tai deque + dict -yhdistelmää

### Matematiikka
- FFT: Toteuta Cooley-Tukey radix-2 algoritmi
- Alkuluvut: Testaa vain parittomilla jakajilla √n:ään asti
- Konveksihulto: Graham scan on tehokas O(n log n)

## Rajoitukset

- **Vain Python standardikirjasto**: Ei NumPy, SciPy, tai muita kolmannen osapuolen kirjastoja
- **Älä muokkaa vertailutiedostoja**: `slow_funcs.py` ja `originals.py` pysyvät muuttumattomina
- **Säilytä API**: Funktiosignatuurien tulee täsmätä täysin
- **Oikeellisuus ensin**: Optimointi ei saa vaarantaa tulosten oikeellisuutta

## Lisähaasteita

Jos haluat mennä pidemmälle:

1. **Profiloi koodisi**: Käytä `cProfile` löytääksesi todellisia pullonkauloja
2. **Vertaa Big O -notaatiota**: Dokumentoi aikavaativuudet ennen ja jälkeen
3. **Kokeile eri syötteitä**: Testaa suurilla ja pienillä syötteillä
4. **Tutki vaihtoehtoja**: Moniin ongelmiin on useita ratkaisuja

## Resurssit

- [Python heapq dokumentaatio](https://docs.python.org/3/library/heapq.html)
- [Python collections dokumentaatio](https://docs.python.org/3/library/collections.html)
- [Algoritmianalyysiopas](https://www.bigocheatsheet.com/)

Onnea edistyneeseen optimointihaasteeseen! 🚀
