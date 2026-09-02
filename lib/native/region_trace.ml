type entry = { key : Vec6.coord; outputs : Vec6.coord list }

type coverage = {
  total : int;
  keys : int;
  visited : int;
  duplicates : int;
  missing : int;
}

type t = {
  program : Region_program.t;
  entries : entry list;
  coverage : coverage;
}

type error =
  [ Region_partition.error
  | `Coverage of coverage
  | `Ownership of Vec6.coord * Vec6.coord ]

let pp_error fmt : [< error ] -> unit = function
  | #Region_partition.error as error -> Region_partition.pp_error fmt error
  | `Coverage { total; keys; visited; duplicates; missing } ->
      Fmt.pf fmt
        "invalid ownership coverage: total=%d keys=%d visited=%d duplicates=%d \
         missing=%d"
        total keys visited duplicates missing
  | `Ownership (output, key) ->
      Fmt.pf fmt "output %a is owned by key %a, not its enumerating key"
        Vec6.pp_coord output Vec6.pp_coord key

let summarize ~output_shape entries =
  let total = (Vec6.numel output_shape :> int) in
  let visits = Array.make total 0 in
  List.iter
    (fun { outputs; _ } ->
      List.iter
        (fun output ->
          let offset = (Vec6.offset output_shape output :> int) in
          visits.(offset) <- visits.(offset) + 1)
        outputs)
    entries;
  let visited, duplicates =
    Array.fold_left
      (fun (visited, duplicates) visits ->
        ( (visited + if visits > 0 then 1 else 0),
          duplicates + if visits > 1 then visits - 1 else 0 ))
      (0, 0) visits
  in
  {
    total;
    keys = List.length entries;
    visited;
    duplicates;
    missing = total - visited;
  }

let collect program ~output_shape =
  Err.Escape.with_escape @@ fun esc ->
  let partition = Region_program.partition program in
  (* [output_shape] has already passed the same bounded shape admission as
     tensor materialisation.  The table is deliberately independent from tensor
     stores so duplicate writes cannot be hidden by a final dense comparison. *)
  let entries =
    Region_partition.fold_keys ~output_shape ~init:[]
      ~f:(fun entries key ->
        let outputs =
          Region_partition.fold_outputs ~output_shape ~key ~init:[]
            ~f:(fun outputs output ->
              let owner =
                Err.Escape.or_throw esc
                  (Err.map_error
                     (fun (error : Region_partition.error) -> (error :> error))
                     (Region_partition.key_of_output ~output_shape partition
                        output))
              in
              if owner <> key then
                Err.Escape.throw esc (`Ownership (output, key))
              else output :: outputs)
            partition
          |> List.rev
        in
        { key; outputs } :: entries)
      partition
    |> List.rev
  in
  let coverage = summarize ~output_shape entries in
  if coverage.duplicates <> 0 || coverage.missing <> 0 then
    Err.Escape.throw esc (`Coverage coverage)
  else { program; entries; coverage }

let pp fmt t =
  let pp_local fmt (local : Region_local.t) =
    Fmt.pf fmt "%a = %a" Expr.Local_var.pp local.id Expr.Pp.value local.value
  in
  let pp_entry fmt { key; outputs } =
    Fmt.pf fmt "key %a@,  locals: %a@,  emit: %a@,  outputs: %a" Vec6.pp_coord
      key
      (Fmt.list ~sep:(Fmt.any "; ") pp_local)
      (Region_program.locals t.program)
      Expr.Pp.value
      (Region_program.output t.program)
      (Fmt.list ~sep:(Fmt.any ",") Vec6.pp_coord)
      outputs
  in
  Fmt.pf fmt
    "program:@,\
     %a@,\
     entries:@,\
     %a@,\
     coverage: total=%d keys=%d visited=%d duplicates=%d missing=%d"
    Region_program.pp t.program
    (Fmt.list ~sep:(Fmt.any "@,") pp_entry)
    t.entries t.coverage.total t.coverage.keys t.coverage.visited
    t.coverage.duplicates t.coverage.missing
