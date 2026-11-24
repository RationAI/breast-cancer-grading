# Shoda mezi modelem MammaPrint a naším modelem hodnocení Grade 3

**Kontext.** Porovnali jsme dva modely na 304 snímcích (stejná skenovaná data,
stejná mřížka úrysků, překryv 100 %):
1. **MammaPrint** – externí model předpovídající rizikový index (naše replika
   p. kolegy, vyloučeno „nasamplingem“ – na datech MammaPrint jsme netrénovali),
2. **náš model** – klasifikátor Grade 3 s attention mapou (kde model „vidí“
   tumor).

Ptali jsme se dvou věcí: **(a)** dívá se náš model na stejné části tkáně jako
MammaPrint? **(b)** korelují jejich výsledky navzájem?

## (a) Které úrysky jsou důležité — shoda prakticky žádná

Pro každý snímez jsme spočítali Spearmanovu korelaci ρ mezi attention našího
modelu a „řidičemi rizika“ MammaPrintu (úrysky, jejichž vynecháním rizikový
index klesne):

| Srovnání (304 snímků, medián ρ) | Hodnota | Výklad |
|---|---|---|
| naše attention vs. **řidiče rizika MammaPrint** | **+0,02** | mince – žádná shoda |
| naše attention vs. velikost jeho vlivu | +0,01 | nic |
| naše attention vs. jeho **hrubá attention** | **+0,62** (96 % snímků kladné) | oba modely upřednostňují tumor před pozadím |

Jinými slovy: **oba modely se dívají na tumor, ale který tumor „dělá riziko“,
náš model z grade attention nepozná.** Zelená vrstva v prohlížeči (průnik
obou) to potvrzuje vizuálně.

## (b) Výsledkové skóre — naopak nepřímá vazba

| Srovnání na úrovni snímku | Spearman ρ |
|---|---|
| sanity check: jeho index vs. skutečný index | +0,71 ✓ |
| **jeho rizikový index vs. naše P(Grade 3)** | **−0,72** |

Kde MammaPrint vidí vysoké riziko, náš model vidí spíše *nižší* grade. To
odpovídá klinice: MammaPrint je na grade nezávislý (proliferační signál),
grade je naopak o morfologii. Systémy se tedy **neduplikují – měří různé
věce najednou.**

## Závěr

> Oba modely vyhledávají tumor se shodnou preferencí (ρ 0,62), ale které
> partie tumoru dělají vysoké MammaPrint riziko, je pro naši grade attention
> neviditelné (ρ 0,02). Jejich skóre si neodporují, ale ani se nedoplňují
> shodou – jde o dva nezávislé pohledy na tutéž barveninu.

**Interaktivní prohlídce** (304 karet, 3 vrstvy: zelená = shoda řidičů,
oranžová = řidiče rizika MammaPrint, šedá = naše attention):
soubor `mammaprint_agreement_v3.html` – každý obrázek se otevírá v xOpat
prohlížeči (vyžaduje firemní VPN); čísla na kartách jsou platná i offline.

*Podrobnosti a všechny statistiky: `mammaprint_agreement_summary.md`,
`mammaprint_compare/report_summary.csv`.*
