open Schema_runtime
open Err.Syntax

type source = { bytes : Safetensors.Bigstring.t; name : string }
type hasher = Safetensors.Bigstring.t -> Pt2_sha256.Digest.t

type verified = {
  memories : (string * Safetensors.Memory.t) list;  (** By file name. *)
  total : int64;
}

type t = {
  storage : Pt2_storage.t String_map.t;
  owned : int64;
  sources : int64;
}

let over_limit what ~limit ~actual =
  Err.fail (`Over_limit { Fault.Over_limit.what; limit; actual })

module String_set = Set.Make (String)

let verify_sources ?(limits = Limits.default) ?(hash = Pt2_sha256.bigstring)
    (doc : Document.t) (sources : source list) =
  let declared =
    List.map (fun (s : Document.Source.t) -> s.pin) doc.checkpoint_files
    @ match doc.graph_owned with Some p -> [ p ] | None -> []
  in
  let supplied = String_set.of_list (List.map (fun s -> s.name) sources) in
  let wanted =
    String_set.of_list (List.map (fun (p : Document.Pin.t) -> p.name) declared)
  in
  let* () =
    match String_set.min_elt_opt (String_set.diff wanted supplied) with
    | Some name -> Err.fail (`Source_missing name)
    | None -> Err.return ()
  in
  let* () =
    match String_set.min_elt_opt (String_set.diff supplied wanted) with
    | Some name -> Err.fail (`Source_surplus name)
    | None -> Err.return ()
  in
  let* () =
    if List.length sources <> String_set.cardinal supplied then
      let rec dup seen = function
        | [] -> assert false
        | s :: rest ->
            if String_set.mem s.name seen then s.name
            else dup (String_set.add s.name seen) rest
      in
      Err.fail
        (`Duplicate (Fault.Checkpoint_file, dup String_set.empty sources))
    else Err.return ()
  in
  let total = ref 0L in
  let* memories =
    Err.List.map
      (fun (pin : Document.Pin.t) ->
        let { bytes; _ } =
          List.find (fun s -> String.equal s.name pin.name) sources
        in
        let actual = Int64.of_int (Bigarray.Array1.dim bytes) in
        let* () =
          if Int64.compare actual limits.max_source_bytes > 0 then
            over_limit Limits.Source_bytes ~limit:limits.max_source_bytes
              ~actual
          else Err.return ()
        in
        let* () =
          if Int64.equal actual pin.size then Err.return ()
          else
            Err.fail
              (`Source_size_mismatch
                 {
                   Fault.Size_clash.target = pin.name;
                   expected = pin.size;
                   actual;
                 })
        in
        let* () =
          let digest = hash bytes in
          if Pt2_sha256.Digest.equal digest pin.sha256 then Err.return ()
          else
            Err.fail
              (`Source_digest_mismatch
                 {
                   Fault.Source_digest.name = pin.name;
                   actual = digest;
                   expected = pin.sha256;
                 })
        in
        total := Int64.add !total actual;
        let+ memory =
          Safetensors.Memory.of_bigstring bytes
          |> Err.import ~pos:__POS__ (fun e ->
              `Safetensors_header (pin.name, Fmt.str "%a" Safetensors.Error.pp e))
        in
        (pin.name, memory))
      declared
  in
  let* () =
    if Int64.compare !total limits.max_prepared_bytes > 0 then
      over_limit Limits.Prepared_bytes ~limit:limits.max_prepared_bytes
        ~actual:!total
    else Err.return ()
  in
  Err.return { memories; total = !total }

let alloc n : Pt2_storage.t =
  Bigarray.Array1.create Bigarray.char Bigarray.c_layout n

(* Budget: reserve [bytes] against the allocation and aggregate ceilings, then
   narrow to [int]. The narrowing is after the bound, which is at most
   [max_allocation_bytes] (a js_of_ocaml-safe ceiling). *)
let reserve (limits : Limits.t) ~sources ~owned bytes =
  if Int64.compare bytes limits.max_allocation_bytes > 0 then
    over_limit Limits.Allocation_bytes ~limit:limits.max_allocation_bytes
      ~actual:bytes
  else
    let after = Int64.add (Int64.add sources !owned) bytes in
    if Int64.compare after limits.max_prepared_bytes > 0 then
      over_limit Limits.Prepared_bytes ~limit:limits.max_prepared_bytes
        ~actual:after
    else begin
      owned := Int64.add !owned bytes;
      Err.return (Int64.to_int bytes)
    end

let fill_pattern (dst : Pt2_storage.t) element =
  let n = Bigarray.Array1.dim dst and w = String.length element in
  if n > 0 then begin
    for j = 0 to min w n - 1 do
      Bigarray.Array1.unsafe_set dst j element.[j]
    done;
    (* Double the filled prefix until the buffer is full. *)
    let filled = ref (min w n) in
    while !filled < n do
      let len = min !filled (n - !filled) in
      Bigarray.Array1.blit
        (Bigarray.Array1.sub dst 0 len)
        (Bigarray.Array1.sub dst !filled len);
      filled := !filled + len
    done
  end

(* A stored tensor of [file]/[key], checked against the dtype and shape the
   conversion expects it to have. *)
let stored memories ~target ~file ~key ~(expect : Dtype.t) ~shape =
  let memory = List.assoc file memories in
  match Safetensors.Index.find (Safetensors.Memory.index memory) key with
  | None -> Err.fail (`Key_missing { Fault.Key_ref.target; file; key })
  | Some tensor ->
      let code =
        Safetensors.Dtype.to_string (Safetensors.Tensor.dtype tensor)
      in
      let* () =
        if String.equal code (Dtype.to_code expect) then Err.return ()
        else
          Err.fail
            (`Stored_dtype
               { Fault.Stored_dtype.target; expected = expect; actual = code })
      in
      let* () =
        if List.equal Int64.equal (Safetensors.Tensor.shape tensor) shape then
          Err.return ()
        else
          Err.fail
            (`Shape_clash
               {
                 Fault.Clash.target;
                 against = Fault.Source;
                 expected = Safetensors.Tensor.shape tensor;
                 actual = shape;
               })
      in
      Safetensors.Memory.tensor_view memory key
      |> Err.import ~pos:__POS__ (fun e ->
          `Safetensors_view (target, Fmt.str "%a" Safetensors.Error.pp e))

let produce limits ~sources ~owned (doc : Document.t) memories target
    (e : Document.Entry.t) : (Pt2_storage.t, [> Fault.error ]) Err.t =
  let bytes = Document.Entry.byte_count e in
  match e.origin with
  | Document.Origin.Empty -> Err.return Pt2_storage.empty
  | Fill element ->
      let* n = reserve limits ~sources ~owned bytes in
      let dst = alloc n in
      fill_pattern dst element;
      Err.return dst
  | Inline raw ->
      let* _ =
        reserve limits ~sources ~owned (Int64.of_int (String.length raw))
      in
      Err.return (Pt2_storage.of_string raw)
  | Pack key ->
      let file = (Option.get doc.graph_owned).Document.Pin.name in
      stored memories ~target ~file ~key ~expect:e.dtype ~shape:e.shape
  | Checkpoint { file; key; convert = Identity; _ } ->
      stored memories ~target ~file ~key ~expect:e.dtype ~shape:e.shape
  | Checkpoint { file; key; convert = Cast { from; to_ = _ }; _ } ->
      let* src =
        stored memories ~target ~file ~key ~expect:from ~shape:e.shape
      in
      let* n = reserve limits ~sources ~owned bytes in
      let dst = alloc n in
      Widen.widen from ~src ~dst;
      Err.return dst

let capture_set ?(limits = Limits.default) ?(hash = Pt2_sha256.bigstring)
    (doc : Document.t) (v : verified) =
  let owned = ref 0L in
  let* entries =
    Err.List.map
      (fun (target, (e : Document.Entry.t)) ->
        let* storage =
          produce limits ~sources:v.total ~owned doc v.memories target e
        in
        let expected = Document.Entry.byte_count e in
        let actual = Int64.of_int (Bigarray.Array1.dim storage) in
        let* () =
          if Int64.equal actual expected then Err.return ()
          else
            Err.fail
              (`Value_length { Fault.Size_clash.target; expected; actual })
        in
        let digest = hash storage in
        let+ () =
          if Pt2_sha256.Digest.equal digest e.sha256 then Err.return ()
          else
            Err.fail
              (`Value_digest_clash
                 {
                   Fault.Clash.target;
                   against = Fault.Computed;
                   expected = digest;
                   actual = e.sha256;
                 })
        in
        (target, storage))
      (String_map.bindings doc.tensors)
  in
  Err.return
    {
      storage = String_map.of_seq (List.to_seq entries);
      owned = !owned;
      sources = v.total;
    }

let find t target = String_map.find_opt target t.storage

let load t target =
  match find t target with
  | Some s -> Ok s
  | None -> Error (Fmt.str "capture %S is not in the prepared set" target)

let targets t = List.map fst (String_map.bindings t.storage)
let owned_bytes t = t.owned
let source_bytes t = t.sources
