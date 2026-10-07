(* The status a selected or physical program's failure record reports: the
   kind word, the invocation word and the payload words read from the record
   region, decoded against the bundle's failure-site table. A record with an
   undefined word is the uninitialized defect; an invocation word of all ones
   is a record whose invocation the program did not know. *)

open Machine_ir

let status memory key ~sites =
  let bytes =
    Mir_memory.read_bytes memory key ~offset:0L
      ~n:(Int64.to_int Mir_failure.record_bytes)
  in
  let word off n =
    let rec go k acc =
      if k < 0 then Some acc
      else
        match bytes.(off + k) with
        | None -> None
        | Some x ->
            go (k - 1) (Int64.logor (Int64.shift_left acc 8) (Int64.of_int x))
    in
    go (n - 1) 0L
  in
  let words =
    List.init Mir_failure.record_words (fun k -> word (8 + (8 * k)) 8)
  in
  match (word 0 4, word 4 4, List.for_all Option.is_some words) with
  | Some kind, Some invocation, true -> (
      let v = Array.of_list (List.map Option.get words) in
      match Mir_failure.decode ~table:sites ~kind:(Int64.to_int32 kind) ~v with
      | Ok (failure, payload, site) ->
          let payload =
            List.map2
              (fun ty bits -> { Mir_const.ty; bits })
              (Mir_failure.payload failure)
              payload
          in
          let invocation =
            if Int64.equal invocation 0xFFFF_FFFFL then None
            else Some (Int64.to_int32 invocation)
          in
          Mir_observation.Status.Failure
            { Mir_observation.Row.failure; payload; invocation; site }
      | Error Mir_failure.Decode_error.Sentinel_site ->
          Mir_observation.Status.Defect Mir_observation.Defect.Sentinel_site
      | Error _ ->
          Mir_observation.Status.Defect Mir_observation.Defect.Invalid_program)
  | _ -> Mir_observation.Status.Defect Mir_observation.Defect.Uninitialized

(* A selected or physical run's status: a returned status word of 0 is
   success; any other reads the record the program stored in [record]. *)
let of_outcome memory binding ~sites ~record (outcome : Mir_interp.Outcome.t) =
  match outcome with
  | Mir_interp.Outcome.Success vs -> (
      match List.rev vs with
      | Mir_datum.Bits 0L :: _ -> Mir_observation.Status.Success
      | Mir_datum.Bits _ :: _ -> (
          match Mir_interp.Binding.instance binding record with
          | Some key -> status memory key ~sites
          | None ->
              Mir_observation.Status.Defect
                Mir_observation.Defect.Invalid_program)
      | _ ->
          Mir_observation.Status.Defect Mir_observation.Defect.Invalid_program)
  | Mir_interp.Outcome.Failure row -> Mir_observation.Status.Failure row
  | Mir_interp.Outcome.Defect (d, _) -> Mir_observation.Status.Defect d
  | Mir_interp.Outcome.Fuel_exhausted -> Mir_observation.Status.Fuel_exhausted
  | Mir_interp.Outcome.Unsupported s -> Mir_observation.Status.Unsupported s
