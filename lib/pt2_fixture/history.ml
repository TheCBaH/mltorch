(* What a static-history decode artifact covers. The producer exports one graph
   per history length: its K/V inputs have a fixed history axis, and a published
   case is a single step at that length. Neither a run nor a replay of it says
   anything about the lengths in between, so a generation loop can use the
   artifact only at exactly the history it was made for, and chaining two
   artifacts is only sound when their shapes meet. These checks are pure and
   refuse, with a reason, rather than pad, truncate or reinterpret. *)

type t = {
  attention_length : int;
  history : int;
  maximum_input_history : int;
  state_inputs : string list;
  state_outputs : string list;
}

type fault =
  | Capacity of { requested : int; maximum : int }
  | Chain_mismatch of { what : string; prefill : string; decode : string }
  | Uncovered of { requested : int; covered : int }

let pp_fault ppf = function
  | Capacity { requested; maximum } ->
      Fmt.pf ppf "history %d exceeds the artifact's capacity of %d" requested
        maximum
  | Chain_mismatch { what; prefill; decode } ->
      Fmt.pf ppf
        "prefill and decode do not meet on %s: prefill has %s, decode %s" what
        prefill decode
  | Uncovered { requested; covered } ->
      Fmt.pf ppf
        "history %d is not covered: this artifact is a static snapshot at \
         history %d, and no other length is established by it"
        requested covered

let scope t =
  Printf.sprintf
    "static snapshot at history %d (attention length %d, capacity %d): other \
     histories are not covered"
    t.history t.attention_length t.maximum_input_history

module Wire = struct
  type variant = { attention_length : int64; history : int64; kind : string }

  type state = {
    input : string list;
    maximum_input_history : int64;
    output : string list;
  }

  type document = { state : state option; variant : variant option }

  let variant_jsont =
    Jsont.Object.map ~kind:"contract variant"
      (fun attention_length history kind -> { attention_length; history; kind })
    |> Jsont.Object.mem "attention_length" Jsont.int64
    |> Jsont.Object.mem "history" Jsont.int64
    |> Jsont.Object.mem "kind" Jsont.string
    |> Jsont.Object.skip_unknown |> Jsont.Object.finish

  let state_jsont =
    Jsont.Object.map ~kind:"contract state"
      (fun input maximum_input_history output ->
        { input; maximum_input_history; output })
    |> Jsont.Object.mem "input" (Jsont.list Jsont.string)
    |> Jsont.Object.mem "maximum_input_history" Jsont.int64
    |> Jsont.Object.mem "output" (Jsont.list Jsont.string)
    |> Jsont.Object.skip_unknown |> Jsont.Object.finish

  let jsont =
    Jsont.Object.map ~kind:"contract history" (fun state variant ->
        { state; variant })
    |> Jsont.Object.opt_mem "state" state_jsont
    |> Jsont.Object.opt_mem "variant" variant_jsont
    |> Jsont.Object.skip_unknown |> Jsont.Object.finish
end

let of_contract_string text =
  match Jsont_bytesrw.decode_string Wire.jsont text with
  | Error e -> Error e
  | Ok { Wire.state = Some s; variant = Some v }
    when String.equal v.kind "static-history" ->
      let unique xs =
        List.sort_uniq String.compare xs |> List.length = List.length xs
      in
      if
        v.history < 0L
        || s.maximum_input_history < v.history
        || s.maximum_input_history > 1_000_000L
        || v.attention_length <= v.history
        || v.attention_length > 1_000_001L
        || s.input = []
        || not (unique s.input && unique s.output)
      then
        Error
          "invalid static history, capacity, attention length or state names"
      else
        Ok
          (Some
             {
               attention_length = Int64.to_int v.attention_length;
               history = Int64.to_int v.history;
               maximum_input_history = Int64.to_int s.maximum_input_history;
               state_inputs = s.input;
               state_outputs = s.output;
             })
  | Ok { Wire.variant = Some v; _ } when v.kind = "static-history" ->
      Error "static-history contract requires state metadata"
  | Ok _ -> Ok None

(* The history axis of a K/V tensor [batch; heads; history; head_dim]. *)
let history_of_shape = function
  | [ b; n; h; d ] when b > 0L && n > 0L && d > 0L && h >= 0L && h <= 1_000_000L
    ->
      Some (Int64.to_int h)
  | _ -> None

let check_feed t ~requested =
  if requested > t.maximum_input_history then
    Error (Capacity { requested; maximum = t.maximum_input_history })
  else if requested <> t.history then
    Error (Uncovered { requested; covered = t.history })
  else Ok ()

(* [prefill] outputs [present_i_key]/[present_i_value] among others; the decode
   artifact's state inputs are [past_i_key]/[past_i_value]. They meet when the
   names correspond one to one, the shapes agree on batch, heads and head
   dimension, and the prefill's length is the decode's history. *)
let chain ~(prefill : (string * int64 list) list) ~(decode : t)
    ~(decode_inputs : (string * int64 list) list) =
  let strip prefix s =
    let n = String.length prefix in
    if String.length s > n && String.equal (String.sub s 0 n) prefix then
      Some (String.sub s n (String.length s - n))
    else None
  in
  let presents =
    List.filter_map
      (fun (name, shape) ->
        Option.map (fun rest -> (rest, shape)) (strip "present_" name))
      prefill
  in
  let pasts =
    List.filter_map
      (fun name ->
        Option.map
          (fun rest -> (rest, List.assoc_opt name decode_inputs))
          (strip "past_" name))
      decode.state_inputs
  in
  let names l = String.concat "," (List.map fst l) in
  if List.map fst presents <> List.map fst pasts then
    Error
      (Chain_mismatch
         {
           what = "state names";
           prefill = names presents;
           decode = names pasts;
         })
  else
    let rec go = function
      | [] -> Ok ()
      | ((rest, p), (_, Some d)) :: tl -> (
          match (p, d) with
          | [ pb; ph; pl; pd ], [ db; dh; dl; dd ] ->
              if pb <> db || ph <> dh || pd <> dd then
                Error
                  (Chain_mismatch
                     {
                       what = "batch, heads or head dimension of " ^ rest;
                       prefill = Printf.sprintf "%Ld,%Ld,%Ld" pb ph pd;
                       decode = Printf.sprintf "%Ld,%Ld,%Ld" db dh dd;
                     })
              else if pl <> Int64.of_int decode.history || pl <> dl then
                Error
                  (Chain_mismatch
                     {
                       what = "history length of " ^ rest;
                       prefill = Int64.to_string pl;
                       decode = string_of_int decode.history;
                     })
              else go tl
          | _ ->
              Error
                (Chain_mismatch
                   {
                     what = "rank of " ^ rest;
                     prefill = "not rank 4";
                     decode = "not rank 4";
                   }))
      | ((rest, _), (_, None)) :: _ ->
          Error
            (Chain_mismatch
               {
                 what = "state input " ^ rest;
                 prefill = "present";
                 decode = "absent";
               })
    in
    go (List.combine presents pasts)

let chain_tensors ~(prefill : Contract.Tensor_spec.t list) ~decode
    ~(decode_inputs : Contract.Tensor_spec.t list) =
  let ( let* ) = Result.bind in
  let shapes =
    List.map (fun (s : Contract.Tensor_spec.t) -> (s.name, s.shape))
  in
  let* () =
    chain ~prefill:(shapes prefill) ~decode
      ~decode_inputs:(shapes decode_inputs)
  in
  let rec check = function
    | [] -> Ok ()
    | name :: _ when not (String.starts_with ~prefix:"past_" name) ->
        Error
          (Chain_mismatch
             { what = "state input name"; prefill = "present_*"; decode = name })
    | name :: rest -> (
        let present = "present_" ^ String.sub name 5 (String.length name - 5) in
        match
          ( List.find_opt
              (fun (s : Contract.Tensor_spec.t) -> s.name = present)
              prefill,
            List.find_opt
              (fun (s : Contract.Tensor_spec.t) -> s.name = name)
              decode_inputs )
        with
        | Some p, Some d when p.dtype = d.dtype -> check rest
        | _ ->
            Error
              (Chain_mismatch
                 {
                   what = "state dtype of " ^ name;
                   prefill = "incompatible";
                   decode = "incompatible";
                 }))
  in
  check decode.state_inputs
