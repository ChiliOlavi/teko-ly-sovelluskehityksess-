# Harjoitus 2 — Virheiden korjaaminen tekoälyllä

Tehtävän tarkoitus

Tässä harjoituksessa sinun tehtäväsi on korjata bugisia Python-funktioita käyttäen kielimallia tai kirjoittamalla koodi itse. Sinulle on annettu kaksi tiedostoa:

- `originals.py` — oikeat, toimivat implementaatiot (vertailuperusta).
- `buggy.py` — virheelliset mutta ajettavat implementaatiot.

Tehtävänanto

1. Luo samaan kansioon tiedosto nimeltä `ratkaisu.py`.
2. Kopioi `buggy.py` sisältö tai kirjoita uudet funktiot siten, että kaikki seuraavat funktiot toimivat kuten `originals.py`-tiedostossa:

   - add
   - multiply_list
   - is_prime
   - factorial
   - fibonacci
   - sum_of_squares
   - reverse_string
   - count_vowels
   - normalize_text
   - compute_stats
   - matrix_multiply
   - evaluate_expression
   - quadratic_roots
   - approximate_root
   - permutations

3. Tallenna muutokset tiedostoon `ratkaisu.py`.
4. Aja grader: `python grader.py` (käyttää Python 3.12). Grader latautuu samasta kansiosta ja vertaa `ratkaisu.py`-funktioita `originals.py`:n versioihin.

Rajoitteet ja vaatimukset

- Käytä Python 3.12:ta.
- Älä lisää ulkoisia kirjastoja — vain standard library on sallittu.
- Säilytä funktioiden allekirjoitukset (parametrit ja paluuarvot) niin, että grader pystyy kutsumaan niitä.
- `ratkaisu.py` ei saa muokata `originals.py`- tai `buggy.py`-tiedostoja.

Arviointi

- Grader suorittaa joukon testisyötteitä jokaiselle funktiolle ja antaa pisteen per funktio, jos kaikki testit kyseiselle funktiolle menevät läpi.
- Maksimipistemäärä on 15.

Vinkkejä

- Aloita helpoimmista funktioista (esim. `add`, `reverse_string`) ja etene monimutkaisempiin (esim. `matrix_multiply`, `approximate_root`).
- Kirjoita pieni testikoodi tai käytä graderia toistuvasti.
- Jos käytät kielimallia, anna sille tarkka tehtävänanto: funktioiden nimet, odotetut parametrit ja paluuarvot.


Onnea matkaan!
