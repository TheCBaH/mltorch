module String_map = Map.Make (String)

type t = {
  vocab : int String_map.t;
  ranks : int String_map.t;  (** "a b" -> rank *)
  suffix : string;
  start_id : int;
  end_id : int;
}

let ( let* ) = Result.bind

let of_json text =
  let* json =
    Jsont_bytesrw.decode_string Jsont.json text |> Result.map_error (fun e -> e)
  in
  let member k = function
    | Jsont.Object (ms, _) ->
        List.find_map (fun ((n, _), v) -> if n = k then Some v else None) ms
    | _ -> None
  in
  match member "model" json with
  | None -> Error "no model"
  | Some model -> (
      match (member "vocab" model, member "merges" model) with
      | Some (Jsont.Object (vs, _)), Some (Jsont.Array (ms, _)) ->
          let vocab =
            List.fold_left
              (fun m ((k, _), v) ->
                match v with
                | Jsont.Number (f, _) -> String_map.add k (int_of_float f) m
                | _ -> m)
              String_map.empty vs
          in
          let ranks, _ =
            List.fold_left
              (fun (m, i) v ->
                match v with
                | Jsont.String (s, _) -> (String_map.add s i m, i + 1)
                | _ -> (m, i + 1))
              (String_map.empty, 0) ms
          in
          let suffix =
            match member "end_of_word_suffix" model with
            | Some (Jsont.String (s, _)) -> s
            | _ -> ""
          in
          let id name =
            Option.to_result ~none:("no token " ^ name)
              (String_map.find_opt name vocab)
          in
          let* start_id = id "<|startoftext|>" in
          let* end_id = id "<|endoftext|>" in
          Ok { vocab; ranks; suffix; start_id; end_id }
      | _ -> Error "no vocab or merges")

let is_letter c = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')
let is_digit c = c >= '0' && c <= '9'

let is_space c =
  c = ' ' || c = '\t' || c = '\n' || c = '\r' || c = '\x0b' || c = '\x0c'

let starts_with s i prefix =
  let n = String.length prefix in
  i + n <= String.length s && String.equal (String.sub s i n) prefix

let contains s needle =
  let n = String.length needle and m = String.length s in
  let rec go i =
    i + n <= m && (String.equal (String.sub s i n) needle || go (i + 1))
  in
  go 0

let contractions = [ "'s"; "'t"; "'re"; "'ve"; "'m"; "'ll"; "'d" ]

let pieces _ text =
  match
    String.to_seq text
    |> Seq.find (fun c ->
        Char.code c >= 128
        || (Char.code c < 32 && not (is_space c))
        || Char.code c = 127)
  with
  | Some c ->
      Error
        (Printf.sprintf "unsupported byte 0x%02x: ASCII text only" (Char.code c))
  | None ->
      if
        contains (String.lowercase_ascii text) "<|startoftext|>"
        || contains (String.lowercase_ascii text) "<|endoftext|>"
      then Error "special-token spellings are refused"
      else
        let s = String.lowercase_ascii text in
        let n = String.length s in
        let out = ref [] in
        let i = ref 0 in
        while !i < n do
          let c = s.[!i] in
          if is_space c then incr i
          else
            match List.find_opt (starts_with s !i) contractions with
            | Some k ->
                out := k :: !out;
                i := !i + String.length k
            | None ->
                let j = ref !i in
                if is_letter c then
                  while !j < n && is_letter s.[!j] do
                    incr j
                  done
                else if is_digit c then incr j
                else
                  while
                    !j < n
                    && (not (is_space s.[!j]))
                    && (not (is_letter s.[!j]))
                    && not (is_digit s.[!j])
                  do
                    incr j
                  done;
                out := String.sub s !i (!j - !i) :: !out;
                i := !j
        done;
        Ok (List.rev !out)

(* Printable ASCII maps to itself under the byte-level alphabet; the piece
   alphabet here never holds anything else (controls are refused). *)
let bpe t piece =
  let chars =
    List.init (String.length piece) (fun i -> String.make 1 piece.[i])
  in
  let symbols =
    match List.rev chars with
    | [] -> []
    | last :: rest -> List.rev ((last ^ t.suffix) :: rest)
  in
  let rec merge symbols =
    let rec pairs = function
      | a :: (b :: _ as rest) -> (a, b) :: pairs rest
      | _ -> []
    in
    let ranked =
      List.filter_map
        (fun (a, b) ->
          Option.map
            (fun r -> (r, a, b))
            (String_map.find_opt (a ^ " " ^ b) t.ranks))
        (pairs symbols)
    in
    match List.sort compare ranked with
    | [] -> symbols
    | (_, a, b) :: _ ->
        let rec go = function
          | x :: y :: rest when String.equal x a && String.equal y b ->
              (a ^ b) :: go rest
          | x :: rest -> x :: go rest
          | [] -> []
        in
        merge (go symbols)
  in
  merge symbols

type encoded = { attention_mask : int list; input_ids : int list }

let encode t ~max_length text =
  let* ps = pieces t text in
  let tokens = List.concat_map (bpe t) ps in
  let ids =
    List.map
      (fun tok ->
        Option.value ~default:t.end_id (String_map.find_opt tok t.vocab))
      tokens
  in
  let keep = max_length - 2 in
  let ids = List.filteri (fun i _ -> i < keep) ids in
  let real = (t.start_id :: ids) @ [ t.end_id ] in
  let n = List.length real in
  Ok
    {
      input_ids = real @ List.init (max_length - n) (fun _ -> t.end_id);
      attention_mask = List.init max_length (fun i -> if i < n then 1 else 0);
    }
