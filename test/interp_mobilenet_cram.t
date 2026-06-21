Run mobilenet_v3_small (the smallest/fastest mobilenet variant) end-to-end
through the graph interpreter on every sample image and compare the local
top-5 against the reference results.json shipped in the release zip, in
--cram mode: when the local top-5 ranking matches the reference, only the
ranking prints (probabilities differ in their low-order digits across
systems, so they'd otherwise make this comparison non-deterministic); on a
mismatch both full rankings print with probabilities at natural precision
for inspection. Per-image inference time goes to stderr, so it is dropped
here too; see `make inference` to see timing and natural-precision
probabilities for any downloaded model. Gated on PT2_DATA so the default
suite needs no download; run via `make pt2.runtest` after
`make pt2.download PT2_MODEL=mobilenet_v3_small`.

  $ ./interp_run.exe "$PT2_DATA/mobilenet_v3_small/mobilenet_v3_small.pt2" "$PT2_DATA/mobilenet_v3_small/images" \
  >   "$PT2_DATA/mobilenet_v3_small/imagenet_lsvrc_2015_synsets.txt" \
  >   "$PT2_DATA/mobilenet_v3_small/imagenet_metadata.txt" "$PT2_DATA/mobilenet_v3_small/results.json" --cram 2>/dev/null
  === images/000000000149.pt ===
  1: n02692877 airship, dirigible
  2: n02782093 balloon
  3: n03888257 parachute, chute
  4: n04486054 triumphal arch
  5: n09229709 bubble
  === images/000000000201.pt ===
  1: n04228054 ski
  2: n04252077 snowmobile
  3: n03218198 dogsled, dog sled, dog sleigh
  4: n04208210 shovel
  5: n09193705 alp
  === images/000000000349.pt ===
  1: n03272562 electric locomotive
  2: n03895866 passenger car, coach, carriage
  3: n04310018 steam locomotive
  4: n04487081 trolleybus, trolley coach, trackless trolley
  5: n04335435 streetcar, tram, tramcar, trolley, trolley car
  === images/000000000389.pt ===
  1: n03630383 lab coat, laboratory coat
  2: n02883205 bow tie, bow-tie, bowtie
  3: n03404251 fur coat
  4: n04325704 stole
  5: n04350905 suit, suit of clothes
  === images/000000000404.pt ===
  1: n03447447 gondola
  2: n03216828 dock, dockage, docking facility
  3: n03662601 lifeboat
  4: n02859443 boathouse
  5: n03947888 pirate, pirate ship
  === images/000000000438.pt ===
  1: n07836838 chocolate sauce, chocolate syrup
  2: n07695742 pretzel
  3: n02776631 bakery, bakeshop, bakehouse
  4: n07693725 bagel, beigel
  5: n13054560 bolete
  === images/000000000564.pt ===
  1: n03032252 cinema, movie theater, movie theatre, movie house, picture palace
  2: n03494278 harmonica, mouth organ, harp, mouth harp
  3: n03529860 home theater, home theatre
  4: n03085013 computer keyboard, keypad
  5: n04296562 stage
  === images/000000000599.pt ===
  1: n02123045 tabby, tabby cat
  2: n02123159 tiger cat
  3: n02124075 Egyptian cat
  4: n03793489 mouse, computer mouse
  5: n04074963 remote control, remote
  === images/000000000605.pt ===
  1: n07920052 espresso
  2: n07930864 cup
  3: n04263257 soup bowl
  4: n02948072 candle, taper, wax light
  5: n04398044 teapot
  === images/000000000612.pt ===
  1: n03388549 four-poster
  2: n04399382 teddy, teddy bear
  3: n03131574 crib, cot
  4: n10148035 groom, bridegroom
  5: n04033995 quilt, comforter, comfort, puff
