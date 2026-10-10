type t = { width : int; height : int; rgb : Bytes.t }

let of_string s =
  let n = String.length s in
  let pos = ref 0 in
  let rec skip_ws () =
    if !pos < n then
      match s.[!pos] with
      | ' ' | '\t' | '\n' | '\r' ->
          incr pos;
          skip_ws ()
      | '#' ->
          while !pos < n && s.[!pos] <> '\n' do
            incr pos
          done;
          skip_ws ()
      | _ -> ()
  in
  let token () =
    skip_ws ();
    let start = !pos in
    while
      !pos < n
      && match s.[!pos] with ' ' | '\t' | '\n' | '\r' -> false | _ -> true
    do
      incr pos
    done;
    String.sub s start (!pos - start)
  in
  let ( let* ) = Result.bind in
  let number what =
    match int_of_string_opt (token ()) with
    | Some v when v > 0 && v <= 65535 -> Ok v
    | _ -> Error (Printf.sprintf "bad PPM %s" what)
  in
  if n < 2 || s.[0] <> 'P' || s.[1] <> '6' then Error "not a binary PPM (P6)"
  else (
    pos := 2;
    let* width = number "width" in
    let* height = number "height" in
    let* maxval = number "maxval" in
    if maxval <> 255 then Error "only maxval 255 is supported"
    else (
      (* Exactly one whitespace byte separates the header from the pixels. *)
      incr pos;
      let need = 3 * width * height in
      if n - !pos < need then Error "truncated PPM pixel data"
      else Ok { width; height; rgb = Bytes.of_string (String.sub s !pos need) }))
