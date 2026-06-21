Run resnet18 (the smallest/fastest resnet variant) end-to-end through the
graph interpreter on every sample image and compare the local top-5 against
the reference results.json shipped in the release zip, in --cram mode: when
the local top-5 ranking matches the reference, only the ranking prints
(probabilities differ in their low-order digits across systems, so they'd
otherwise make this comparison non-deterministic); on a mismatch both full
rankings print with probabilities at natural precision for inspection.
Per-image inference time goes to stderr, so it is dropped here too; see
`make inference` to see timing and natural-precision probabilities for any
downloaded model. Gated on PT2_DATA so the default suite needs no download;
run via `make pt2.runtest` after `make pt2.download`.

  $ ./interp_run.exe "$PT2_DATA/resnet18/resnet18.pt2" "$PT2_DATA/resnet18/images" \
  >   "$PT2_DATA/resnet18/imagenet_lsvrc_2015_synsets.txt" \
  >   "$PT2_DATA/resnet18/imagenet_metadata.txt" "$PT2_DATA/resnet18/results.json" --cram 2>/dev/null
  === images/000000000149.pt ===
  1: n03888257 parachute, chute
  2: n03355925 flagpole, flagstaff
  3: n02782093 balloon
  4: n03445777 golf ball
  5: n04254680 soccer ball
  === images/000000000201.pt ===
  1: n04228054 ski
  2: n04252077 snowmobile
  3: n04208210 shovel
  4: n03218198 dogsled, dog sled, dog sleigh
  5: n02966687 carpenter's kit, tool kit
  === images/000000000349.pt ===
  1: n03272562 electric locomotive
  2: n03895866 passenger car, coach, carriage
  3: n04310018 steam locomotive
  4: n03393912 freight car
  5: n03776460 mobile home, manufactured home
  === images/000000000389.pt ===
  1: n04532106 vestment
  2: n03630383 lab coat, laboratory coat
  3: n03404251 fur coat
  4: n04591157 Windsor tie
  5: n02883205 bow tie, bow-tie, bowtie
  === images/000000000404.pt ===
  1: n03216828 dock, dockage, docking facility
  2: n03662601 lifeboat
  3: n02859443 boathouse
  4: n09332890 lakeside, lakeshore
  5: n04612504 yawl
  === images/000000000438.pt ===
  1: n07695742 pretzel
  2: n02776631 bakery, bakeshop, bakehouse
  3: n07693725 bagel, beigel
  4: n07836838 chocolate sauce, chocolate syrup
  5: n07711569 mashed potato
  === images/000000000564.pt ===
  1: n03032252 cinema, movie theater, movie theatre, movie house, picture palace
  2: n04418357 theater curtain, theatre curtain
  3: n02802426 basketball
  4: n03255030 dumbbell
  5: n04296562 stage
  === images/000000000599.pt ===
  1: n02123045 tabby, tabby cat
  2: n02123159 tiger cat
  3: n02124075 Egyptian cat
  4: n03793489 mouse, computer mouse
  5: n04074963 remote control, remote
  === images/000000000605.pt ===
  1: n04263257 soup bowl
  2: n07930864 cup
  3: n02948072 candle, taper, wax light
  4: n04476259 tray
  5: n03775546 mixing bowl
  === images/000000000612.pt ===
  1: n04429376 throne
  2: n03026506 Christmas stocking
  3: n04399382 teddy, teddy bear
  4: n03388549 four-poster
  5: n02105641 Old English sheepdog, bobtail
