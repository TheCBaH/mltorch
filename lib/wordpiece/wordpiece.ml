module String_map = Map.Make (String)

type vocab = { ids : int String_map.t; tokens : string array }

let vocab_of_lines lines =
  let tokens = Array.of_list lines in
  let ids, _ =
    List.fold_left
      (fun (m, i) t ->
        (* The first occurrence of a repeated token keeps its id, as a
           dictionary built in file order does. *)
        ((if String_map.mem t m then m else String_map.add t i m), i + 1))
      (String_map.empty, 0) lines
  in
  { ids; tokens }

let size v = Array.length v.tokens
let token v i = if i >= 0 && i < size v then Some v.tokens.(i) else None
let id v t = String_map.find_opt t v.ids

type error =
  | Non_ascii of char
  | Special_token_spelling of string
  | Missing_special of string

let pp_error ppf = function
  | Non_ascii c ->
      Fmt.pf ppf "non-ASCII byte 0x%02x: only ASCII text is supported"
        (Char.code c)
  | Special_token_spelling s ->
      Fmt.pf ppf
        "%S is a special-token spelling a reference tokenizer keeps whole; it \
         is refused here"
        s
  | Missing_special s -> Fmt.pf ppf "the vocabulary has no %s token" s

let specials = [ "[CLS]"; "[SEP]"; "[PAD]"; "[MASK]"; "[UNK]" ]

let contains_ci text needle =
  let t = String.uppercase_ascii text and n = String.uppercase_ascii needle in
  let lt = String.length t and ln = String.length n in
  let rec go i =
    i + ln <= lt && (String.equal (String.sub t i ln) n || go (i + 1))
  in
  go 0

let is_whitespace c = c = ' ' || c = '\t' || c = '\n' || c = '\r'

let is_control c =
  (* Control characters other than the whitespace ones are dropped. *)
  let k = Char.code c in
  (k < 32 || k = 127) && not (is_whitespace c)

(* BERT's [_is_punctuation] on ASCII: every non-alphanumeric printable. *)
let is_punctuation c =
  let k = Char.code c in
  (k >= 33 && k <= 47)
  || (k >= 58 && k <= 64)
  || (k >= 91 && k <= 96)
  || (k >= 123 && k <= 126)

let words text =
  match String.to_seq text |> Seq.find (fun c -> Char.code c >= 128) with
  | Some c -> Error (Non_ascii c)
  | None -> (
      match List.find_opt (contains_ci text) specials with
      | Some s -> Error (Special_token_spelling s)
      | None ->
          let out = ref [] and cur = Buffer.create 16 in
          let flush () =
            if Buffer.length cur > 0 then (
              out := Buffer.contents cur :: !out;
              Buffer.clear cur)
          in
          String.iter
            (fun c ->
              if is_control c then ()
              else if is_whitespace c then flush ()
              else if is_punctuation c then (
                flush ();
                out := String.make 1 c :: !out)
              else Buffer.add_char cur (Char.lowercase_ascii c))
            text;
          flush ();
          Ok (List.rev !out))

let max_word_chars = 100

let pieces vocab word =
  if String.length word > max_word_chars then [ "[UNK]" ]
  else
    let n = String.length word in
    let rec go start acc =
      if start >= n then Some (List.rev acc)
      else
        let rec longest stop =
          if stop <= start then None
          else
            let sub = String.sub word start (stop - start) in
            let sub = if start > 0 then "##" ^ sub else sub in
            if String_map.mem sub vocab.ids then Some (sub, stop)
            else longest (stop - 1)
        in
        match longest n with
        | None -> None
        | Some (piece, stop) -> go stop (piece :: acc)
    in
    match go 0 [] with Some p -> p | None -> [ "[UNK]" ]

type encoded = {
  attention_mask : int list;
  input_ids : int list;
  tokens : string list;
  token_type_ids : int list;
}

let encode vocab ~max_length text =
  let ( let* ) = Result.bind in
  let need name =
    match id vocab name with
    | Some i -> Ok i
    | None -> Error (Missing_special name)
  in
  let* cls = need "[CLS]" in
  let* sep = need "[SEP]" in
  let* pad = need "[PAD]" in
  let* unk = need "[UNK]" in
  let* ws = words text in
  let body = List.concat_map (pieces vocab) ws in
  let keep = max_length - 2 in
  let body = List.filteri (fun i _ -> i < keep) body in
  let ids = List.map (fun t -> Option.value ~default:unk (id vocab t)) body in
  let real = [ cls ] @ ids @ [ sep ] in
  let n_real = List.length real in
  let padding = List.init (max_length - n_real) (fun _ -> pad) in
  let tokens =
    List.map
      (fun i -> Option.value ~default:"[UNK]" (token vocab i))
      (real @ padding)
  in
  Ok
    {
      attention_mask =
        List.init max_length (fun i -> if i < n_real then 1 else 0);
      input_ids = real @ padding;
      tokens;
      token_type_ids = List.init max_length (fun _ -> 0);
    }
