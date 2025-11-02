# Harjoitus 3 — Koodin optimointi tekoälyllä

Tavoite
--------
Tässä harjoituksessa tehtävänäsi on kirjoittaa parempi, nopeampi versio annetuista hitaista funktioista käyttäen Python 3.12:ta. Käytä koodin parannukseen mieluiten koodin refaktorointia ja algoritmisten ratkaisujen parantamista. Älä tuo ulkoisia kirjastoja; käytä vain Pythonin standardikirjastoa.

Mitä paketissa on
-----------------
- `originals.py` — nopeasti toteutetut viiteversiot (oikeat tulokset, tehokkaita).
- `slow_funcs.py` — tarkoituksella hitaat versiot samoille funktioille. Näiden kanssa sinun tulee kilpailla suorituskyvyssä.
- `ratkaisu_template.py` — käynnistuspohja, johon voit kirjoittaa oman `ratkaisu.py`-tiedoston.
- `Tehtävänanto.md` — tämä tiedosto.
- `speed_compare.py` — skripti, joka vertaa `slow_funcs`- ja sinun `ratkaisu.py`-funktioidesi nopeutta.

Tehtäväsi
---------
1. Luo tiedosto `ratkaisu.py` samaan kansioon (`Harjoitus_3`).
2. Kopioi funktioiden nimet ja rajapinnat `slow_funcs.py`- tai `originals.py`-tiedostoista ja implementoi ne nopeammin kuin `slow_funcs.py`.
3. Varmista, että kaikilla funktioilla on sama signatuuri ja että ne palauttavat samat arvot kuin `originals.py` (tai `slow_funcs.py`) odottaa.
4. Aja `speed_compare.py` ja pyri saamaan suurempi nopeus (pienempi aikavirta) useimmissa funktioissa verrattuna `slow_funcs.py`.

Arviointi ja suoritus
---------------------
- Suorita vertailu:

```pwsh
python speed_compare.py
```

- Skripti kertoo per-funktio suoritusaikatiedot ja nopeussuhteen (speedup). Tavoite on, että sinun `ratkaisu.py` on selvästi nopeampi kuin `slow_funcs.py` useimmissa testeissä.

Vinkkejä
-------
- Testaa ensin pienillä syötteillä. Varmista oikeellisuus ennen optimointia.
- Paranna algoritmin aikavaativuutta (esim. O(n^2) -> O(n log n) tai vähemmän) aina kun mahdollista.
- Vältä turhia kopioita, toistuvia laskelmia ja raskaita rekursioita (ellei niillä ole muistietua).

Rajoitukset
-----------
- Käytä vain Pythonin standardikirjastoa.
- Älä muokkaa `slow_funcs.py` tai `originals.py` — ne ovat vertailua varten.

Onnea harjoitukseen!
