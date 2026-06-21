Run vit_b_32 (the smallest/fastest vit variant, ~1.45s/image vs ~3.7s for
vit_b_16's finer 16x16 patch grid) end-to-end through the graph interpreter
on every sample image and compare the local top-5 against the reference
results.json shipped in the release zip, in --cram mode: when the local
top-5 ranking matches the reference, only the ranking prints (probabilities
differ in their low-order digits across systems, so they'd otherwise make
this comparison non-deterministic); on a mismatch both full rankings print
with probabilities at natural precision for inspection. Per-image inference
time goes to stderr, so it is dropped here too; see `make inference` to see
timing and natural-precision probabilities for any downloaded model. Gated
on PT2_DATA so the default suite needs no download; run via
`make pt2.runtest` after `make pt2.download PT2_MODEL=vit_b_32`.

  $ ./interp_run.exe "$PT2_DATA/vit_b_32/vit_b_32.pt2" "$PT2_DATA/vit_b_32/images" \
  >   "$PT2_DATA/vit_b_32/imagenet_lsvrc_2015_synsets.txt" \
  >   "$PT2_DATA/vit_b_32/imagenet_metadata.txt" "$PT2_DATA/vit_b_32/results.json" --cram 2>/dev/null
  === images/000000000149.pt ===
  1: n03888257 parachute, chute
  2: n02782093 balloon
  3: n03355925 flagpole, flagstaff
  4: n04254680 soccer ball
  5: n02692877 airship, dirigible
  === images/000000000201.pt ===
  1: n04228054 ski
  2: n03218198 dogsled, dog sled, dog sleigh
  3: n04548362 wallet, billfold, notecase, pocketbook
  4: n02840245 binder, ring-binder
  5: n04252077 snowmobile
  === images/000000000349.pt ===
  1: n03895866 passenger car, coach, carriage
  2: n03272562 electric locomotive
  3: n03393912 freight car
  4: n02917067 bullet train, bullet
  5: n04335435 streetcar, tram, tramcar, trolley, trolley car
  === images/000000000389.pt ===
  1: n03630383 lab coat, laboratory coat
  2: n04591157 Windsor tie
  3: n04532106 vestment
  4: n04317175 stethoscope
  5: n02883205 bow tie, bow-tie, bowtie
  === images/000000000404.pt ===
  1: n03662601 lifeboat
  2: n03216828 dock, dockage, docking facility
  3: n04612504 yawl
  4: n02859443 boathouse
  5: n09428293 seashore, coast, seacoast, sea-coast
  === images/000000000438.pt ===
  1: n02776631 bakery, bakeshop, bakehouse
  2: n07695742 pretzel
  3: n07684084 French loaf
  4: n04476259 tray
  5: n07836838 chocolate sauce, chocolate syrup
  === images/000000000564.pt ===
  1: n02777292 balance beam, beam
  2: n04039381 racket, racquet
  3: n04296562 stage
  4: n03032252 cinema, movie theater, movie theatre, movie house, picture palace
  5: n03535780 horizontal bar, high bar
  === images/000000000599.pt ===
  1: n02123045 tabby, tabby cat
  2: n02123159 tiger cat
  3: n02124075 Egyptian cat
  4: n04074963 remote control, remote
  5: n03793489 mouse, computer mouse
  === images/000000000605.pt ===
  1: n07920052 espresso
  2: n07930864 cup
  3: n02948072 candle, taper, wax light
  4: n04476259 tray
  5: n04263257 soup bowl
  === images/000000000612.pt ===
  1: n04429376 throne
  2: n03388549 four-poster
  3: n04399382 teddy, teddy bear
  4: n03447721 gong, tam-tam
  5: n02699494 altar
