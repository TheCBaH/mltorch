(* Running a hand-built generic program: regions declared and bound, pointer
   arguments made from views, and binary32/binary64 cells packed into and read
   back from bytes independently of the interpreter. *)
open Machine_ir
open Machine_interp

let region id size =
  {
    Mir_region.id = Mir_id.Region.of_int id;
    size;
    align = 16L;
    init = Mir_region.Bound;
  }

let view ?(perm = Mir_view.Read_write) ?(role = Mir_view.Input) ?(offset = 0L)
    id ~region size =
  {
    Mir_view.id = Mir_id.View.of_int id;
    region = Mir_id.Region.of_int region;
    offset;
    size;
    perm;
    role;
    source = None;
  }

let with_objects (p : Mir_program.generic) ~regions ~views =
  { p with Mir_program.regions; views }

let le_bytes n bits =
  String.init n (fun k ->
      Char.chr
        (Int64.to_int
           (Int64.logand (Int64.shift_right_logical bits (8 * k)) 0xFFL)))

let f32_bytes xs =
  String.concat ""
    (List.map (fun x -> le_bytes 4 (Int64.of_int32 (Int32.bits_of_float x))) xs)

let verify p =
  match Err.payload (Mir_verify.generic p) with
  | Ok g -> g
  | Error d -> Fmt.failwith "%a" Mir_diagnostic.pp d

(* Runs [p] with [bound] bytes per region id and [args]; [ptr k] in [args]
   means a pointer to view [k]. *)
type arg = Ptr of int | Val of Mir_datum.t

let run ?fuel ?models ?(bound = []) p args =
  let g = verify p in
  let memory = Mir_memory.create () in
  match
    Mir_interp.instantiate p memory ~bound:(fun r ->
        List.assoc_opt (Mir_id.Region.to_int r) bound)
  with
  | Error e -> failwith e
  | Ok binding ->
      let args =
        List.map
          (function
            | Val v -> v
            | Ptr k -> (
                match
                  Mir_interp.Binding.view binding p (Mir_id.View.of_int k)
                with
                | Some ptr -> Mir_datum.Ptr ptr
                | None -> failwith "no such view"))
          args
      in
      let r = Mir_interp.run ?fuel ?models g memory binding ~args in
      (r, memory, binding)

let show_outcome (r : Mir_interp.run) =
  let events =
    List.filter_map
      (fun (e, n) ->
        if Int64.equal n 0L then None
        else Some (Fmt.str "%s=%Ld" (Mir_event.name e) n))
      r.Mir_interp.events
  in
  Fmt.str "%a%s" Mir_interp.Outcome.pp r.Mir_interp.outcome
    (if events = [] then "" else " events " ^ String.concat "," events)

let f64_result (r : Mir_interp.run) =
  match r.Mir_interp.outcome with
  | Mir_interp.Outcome.Success [ Mir_datum.Bits b ] ->
      Some (Int64.float_of_bits b)
  | _ -> None

(* The observation a run makes: status and the given scalar results as a
   pseudo-output, plus events. *)
let observe ?(source = Expr.Source.create 0) (r : Mir_interp.run) ~results =
  let status, cells =
    match r.Mir_interp.outcome with
    | Mir_interp.Outcome.Success vs ->
        ( Mir_observation.Status.Success,
          Array.of_list
            (List.map2
               (fun ty v ->
                 match v with
                 | Mir_datum.Bits bits -> Some { Mir_const.ty; bits }
                 | Mir_datum.Flags _ | Mir_datum.Lanes _ | Mir_datum.Order
                 | Mir_datum.Ptr _ ->
                     None)
               results vs) )
    | Mir_interp.Outcome.Failure row ->
        (Mir_observation.Status.Failure row, [||])
    | Mir_interp.Outcome.Defect (d, _) -> (Mir_observation.Status.Defect d, [||])
    | Mir_interp.Outcome.Fuel_exhausted ->
        (Mir_observation.Status.Fuel_exhausted, [||])
    | Mir_interp.Outcome.Unsupported s ->
        (Mir_observation.Status.Unsupported s, [||])
  in
  {
    Mir_observation.status;
    outputs =
      (match status with
      | Mir_observation.Status.Success ->
          [ { Mir_observation.Output.source; cells } ]
      | _ -> []);
    events = r.Mir_interp.events;
  }

let verdict ~expected ~actual =
  match Mir_compare.observations ~expected ~actual () with
  | Ok () -> "agree"
  | Error d -> Fmt.str "mismatch: %a" Mir_compare.Difference.pp d
