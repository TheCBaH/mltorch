(* CLIP's input side without the real assets: Pillow's resample pinned by
   checksums of its output on deterministic images, the PPM reader, the shorter-
   edge sizes, and the BPE on a small tokenizer.json written by hand. *)

open Clip_input

(* A deterministic RGB image: the low bits of a linear congruential stream. *)
let image h w seed =
  let s = ref seed and b = Bytes.create (h * w * 3) in
  Bytes.iteri
    (fun i _ ->
      s := ((!s * 1103515245) + 12345) land 0x7fffffff;
      Bytes.set b i (Char.chr ((!s lsr 16) land 255)))
    b;
  { Ppm.width = w; height = h; rgb = b }

let checksum (img : Ppm.t) =
  Bytes.fold_left
    (fun h c -> ((h * 31) + Char.code c) land 0xffffffff)
    0 img.rgb

(* The checksums are of what Pillow's [Image.resize(..., Image.BICUBIC)] returns
   for the same bytes: upscales, downscales, one axis only, and larger shapes. *)
let%expect_test "bicubic resize equals Pillow's, byte for byte" =
  List.iter
    (fun (h, w, oh, ow, expected) ->
      let out =
        Resample.bicubic (image h w ((h * 1000) + w)) ~width:ow ~height:oh
      in
      Printf.printf "%dx%d -> %dx%d: %s\n" h w oh ow
        (if checksum out = expected && out.width = ow && out.height = oh then
           "equal"
         else Printf.sprintf "DIFFERENT (%d)" (checksum out)))
    [
      (6, 8, 3, 4, 3498744994);
      (5, 5, 9, 9, 3363784255);
      (7, 3, 7, 5, 2794800373);
      (4, 10, 2, 5, 1291808257);
      (90, 160, 224, 398, 992406838);
      (300, 200, 224, 336, 1170237086);
    ];
  [%expect
    {|
    6x8 -> 3x4: equal
    5x5 -> 9x9: equal
    7x3 -> 7x5: equal
    4x10 -> 2x5: equal
    90x160 -> 224x398: equal
    300x200 -> 224x336: equal |}]

let%expect_test
    "an unchanged size is not resampled; a constant image stays constant" =
  let img = image 5 7 3 in
  let same = Resample.bicubic img ~width:7 ~height:5 in
  Printf.printf "identity: %b\n" (Bytes.equal same.rgb img.rgb);
  let flat =
    { Ppm.width = 6; height = 6; rgb = Bytes.make (6 * 6 * 3) '\x80' }
  in
  let out = Resample.bicubic flat ~width:4 ~height:9 in
  Printf.printf "flat: %b\n" (Bytes.for_all (fun c -> c = '\x80') out.rgb);
  [%expect {|
    identity: true
    flat: true |}]

let%expect_test "the shorter edge becomes the size; the longer is truncated" =
  List.iter
    (fun (w, h) ->
      let a, b = Prep.resized_size ~width:w ~height:h ~size:224 in
      Printf.printf "%dx%d -> %dx%d\n" w h a b)
    [ (400, 300); (300, 400); (224, 224); (1000, 700); (331, 225); (91, 57) ];
  [%expect
    {|
    400x300 -> 298x224
    300x400 -> 224x298
    224x224 -> 224x224
    1000x700 -> 320x224
    331x225 -> 329x224
    91x57 -> 357x224 |}]

let%expect_test "PPM: comments, one separator byte, and refusals" =
  let ok = "P6\n# a comment\n2 1\n255\n\001\002\003\004\005\006" in
  (match Ppm.of_string ok with
  | Ok i -> Printf.printf "%dx%d %S\n" i.width i.height (Bytes.to_string i.rgb)
  | Error e -> print_endline e);
  List.iter
    (fun s ->
      match Ppm.of_string s with
      | Ok _ -> print_endline "ok"
      | Error e -> print_endline e)
    [
      "P5\n1 1\n255\n\000";
      "P6\n1 1\n65535\n\000\000\000";
      "P6\n2 2\n255\n\000";
      "P6\nx 1\n255\n";
    ];
  [%expect
    {|
    2x1 "\001\002\003\004\005\006"
    not a binary PPM (P6)
    only maxval 255 is supported
    truncated PPM pixel data
    bad PPM width |}]

(* A tokenizer.json with just what the BPE reads: a vocabulary with the byte
   alphabet it uses, a few merges (ranked by order) and the </w> suffix. *)
let tokenizer =
  {|{"model":{"type":"BPE","end_of_word_suffix":"</w>","vocab":{"a":0,"b":1,"c":2,"t":3,"h":4,"e":5,"s":6,"'":7,"1":8,"!":9,"a</w>":10,"b</w>":11,"c</w>":12,"t</w>":13,"h</w>":14,"e</w>":15,"s</w>":16,"'</w>":17,"1</w>":18,"!</w>":19,"th":20,"the</w>":21,"ab":22,"abc</w>":23,"'s</w>":24,"he</w>":25,"<|startoftext|>":26,"<|endoftext|>":27},"merges":["t h","h e</w>","th e</w>","a b","ab c</w>","' s</w>"]}}|}

let show label text =
  match Bpe.of_json tokenizer with
  | Error e -> print_endline e
  | Ok t -> (
      match Bpe.encode t ~max_length:8 text with
      | Ok e ->
          Printf.printf "%-22s ids %s | mask %s\n" label
            (String.concat "," (List.map string_of_int e.input_ids))
            (String.concat "" (List.map string_of_int e.attention_mask))
      | Error e -> Printf.printf "%-22s %s\n" label e)

let%expect_test "BPE: merges by rank, the end-of-word suffix, the pattern" =
  (* "the" -> t h e</w> -> th e</w> -> the</w> (21); "abc" -> ab c</w> -> abc</w>. *)
  show "the abc" "The ABC";
  (* A contraction is its own piece: "cat's" is "c a t" + "'s". *)
  show "contraction" "a's";
  (* Punctuation runs and single digits are separate pieces. *)
  show "digits, punctuation" "a1!1";
  show "empty" "";
  show "truncated" "a b c a b c a b c";
  [%expect
    {|
    the abc                ids 26,21,23,27,27,27,27,27 | mask 11110000
    contraction            ids 26,10,24,27,27,27,27,27 | mask 11110000
    digits, punctuation    ids 26,10,18,19,18,27,27,27 | mask 11111100
    empty                  ids 26,27,27,27,27,27,27,27 | mask 11000000
    truncated              ids 26,10,11,12,10,11,12,27 | mask 11111111 |}]

let%expect_test "BPE: what is refused" =
  show "non-ASCII" "caf\xc3\xa9";
  show "control" "a\x01b";
  show "special spelling" "x <|endoftext|>";
  [%expect
    {|
    non-ASCII              unsupported byte 0xc3: ASCII text only
    control                unsupported byte 0x01: ASCII text only
    special spelling       special-token spellings are refused |}]
