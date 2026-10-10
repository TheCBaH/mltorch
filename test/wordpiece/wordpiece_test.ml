(* The tokenizer on a small vocabulary, worked out by hand from BERT's
   documented algorithm: lower-casing, punctuation as its own word, greedy
   longest match with a [##] continuation, [[UNK]] for a word with an unmatched
   part, truncation and padding. *)

let vocab =
  Wordpiece.vocab_of_lines
    [
      "[PAD]";
      "[UNK]";
      "[CLS]";
      "[SEP]";
      "hello";
      "world";
      "un";
      "##aff";
      "##able";
      "##a";
      "the";
      "cat";
      ",";
      ".";
      "!";
      "don";
      "'";
      "t";
      "a";
      "##b";
      "##ab";
      "##abc";
    ]

let show label text ~max_length =
  match Wordpiece.encode vocab ~max_length text with
  | Error e -> Fmt.pr "%s: %a@." label Wordpiece.pp_error e
  | Ok e ->
      Fmt.pr "%s: %s | ids %s | mask %s@." label
        (String.concat " " e.tokens)
        (String.concat "," (List.map string_of_int e.input_ids))
        (String.concat "" (List.map string_of_int e.attention_mask))

let%expect_test "words, punctuation and case" =
  show "plain" "Hello, WORLD!" ~max_length:8;
  show "apostrophe" "don't" ~max_length:8;
  show "whitespace and control" "  the\tcat\n\x01 " ~max_length:6;
  [%expect
    {|
    plain: [CLS] hello , world ! [SEP] [PAD] [PAD] | ids 2,4,12,5,14,3,0,0 | mask 11111100
    apostrophe: [CLS] don ' t [SEP] [PAD] [PAD] [PAD] | ids 2,15,16,17,3,0,0,0 | mask 11111000
    whitespace and control: [CLS] the cat [SEP] [PAD] [PAD] | ids 2,10,11,3,0,0 | mask 111100 |}]

let%expect_test "wordpiece: longest match, continuation, and [UNK]" =
  show "unaffable" "unaffable" ~max_length:6;
  (* "a" then the longest "##abc" over "##ab"+"##c"; the greedy longest wins. *)
  show "longest first" "aabc" ~max_length:6;
  (* "ab" has the piece "a" followed by "##b". *)
  show "two pieces" "ab" ~max_length:5;
  (* "unxyz": "un" matches but "##xyz" has no piece, so the whole word is [UNK]. *)
  show "unmatched part" "unxyz" ~max_length:5;
  show "unknown word" "zzz" ~max_length:4;
  [%expect
    {|
    unaffable: [CLS] un ##aff ##able [SEP] [PAD] | ids 2,6,7,8,3,0 | mask 111110
    longest first: [CLS] a ##abc [SEP] [PAD] [PAD] | ids 2,18,21,3,0,0 | mask 111100
    two pieces: [CLS] a ##b [SEP] [PAD] | ids 2,18,19,3,0 | mask 11110
    unmatched part: [CLS] [UNK] [SEP] [PAD] [PAD] | ids 2,1,3,0,0 | mask 11100
    unknown word: [CLS] [UNK] [SEP] [PAD] | ids 2,1,3,0 | mask 1110 |}]

let%expect_test
    "truncation keeps the specials; a word over 100 characters is [UNK]" =
  show "truncated" "the cat the cat the cat" ~max_length:5;
  show "exactly full" "the cat" ~max_length:4;
  (* "a" then ninety-nine "##a" pieces fit the vocabulary, so a word that is
     matched at the limit is distinguished from one rejected for its length. *)
  show "101 characters is rejected" (String.make 101 'a') ~max_length:4;
  (match Wordpiece.encode vocab ~max_length:105 (String.make 100 'a') with
  | Ok e ->
      Fmt.pr "100 characters are matched: %d pieces@."
        (List.fold_left ( + ) 0 e.attention_mask - 2)
  | Error _ -> print_endline "error");
  [%expect
    {|
    truncated: [CLS] the cat the [SEP] | ids 2,10,11,10,3 | mask 11111
    exactly full: [CLS] the cat [SEP] | ids 2,10,11,3 | mask 1111
    101 characters is rejected: [CLS] [UNK] [SEP] [PAD] | ids 2,1,3,0 | mask 1110
    100 characters are matched: 100 pieces |}]

let%expect_test "what is refused rather than guessed" =
  show "non-ASCII" "caf\xc3\xa9" ~max_length:6;
  show "special spelling" "hello [SEP] world" ~max_length:8;
  show "special spelling, other case" "the [cls]" ~max_length:8;
  (match
     Wordpiece.encode
       (Wordpiece.vocab_of_lines [ "[PAD]"; "[CLS]"; "[SEP]" ])
       ~max_length:4 "x"
   with
  | Error e -> Fmt.pr "no [UNK]: %a@." Wordpiece.pp_error e
  | Ok _ -> print_endline "ran");
  [%expect
    {|
    non-ASCII: non-ASCII byte 0xc3: only ASCII text is supported
    special spelling: "[SEP]" is a special-token spelling a reference tokenizer keeps whole; it is refused here
    special spelling, other case: "[CLS]" is a special-token spelling a reference tokenizer keeps whole; it is refused here
    no [UNK]: the vocabulary has no [UNK] token |}]

(* Which ASCII characters split a word: the four punctuation ranges, with the
   characters just outside each (digits, letters, DEL) left in. Split, "the" "/"
   "cat" is three tokens ("/" is [UNK]); unsplit, "the/cat" is one [UNK]. *)
let%expect_test "the punctuation ranges, at their edges" =
  let cases =
    [
      '/';
      '0';
      ':';
      '@';
      'A';
      '[';
      '\\';
      ']';
      '^';
      '_';
      '`';
      'a';
      '{';
      '~';
      '\x7f';
    ]
  in
  List.iter
    (fun c ->
      let text = Printf.sprintf "the%ccat" c in
      match Wordpiece.words text with
      | Ok w -> Fmt.pr "%S -> %d word(s)@." text (List.length w)
      | Error e -> Fmt.pr "%S -> %a@." text Wordpiece.pp_error e)
    cases;
  [%expect
    {|
    "the/cat" -> 3 word(s)
    "the0cat" -> 1 word(s)
    "the:cat" -> 3 word(s)
    "the@cat" -> 3 word(s)
    "theAcat" -> 1 word(s)
    "the[cat" -> 3 word(s)
    "the\\cat" -> 3 word(s)
    "the]cat" -> 3 word(s)
    "the^cat" -> 3 word(s)
    "the_cat" -> 3 word(s)
    "the`cat" -> 3 word(s)
    "theacat" -> 1 word(s)
    "the{cat" -> 3 word(s)
    "the~cat" -> 3 word(s)
    "the\127cat" -> 1 word(s) |}]
