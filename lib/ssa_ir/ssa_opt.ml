type pass = { name : string; run : Ssa_program.t -> Ssa_program.t * bool }

let simplify = { name = "simplify"; run = Ssa_opt_simplify.run }
let guards = { name = "guards"; run = Ssa_opt_guards.pass }
let hoist ~alias = { name = "hoist"; run = Ssa_opt_hoist.pass ~alias }
let share ~alias = { name = "share"; run = Ssa_opt_share.pass ~alias }

let block ~alias ~group =
  { name = "block"; run = Ssa_opt_block.pass ~policy:alias ~group }

let vectorize ~alias ~target =
  {
    name = "vectorize";
    run =
      (fun p ->
        let q, report = Ssa_vectorize.program ~alias ~target p in
        let changed =
          List.exists
            (fun (d : Ssa_vectorize.Decision.t) ->
              d.Ssa_vectorize.Decision.outcome
              = Ssa_vectorize.Decision.Vectorized)
            report
        in
        ((if changed then q else p), changed));
  }

let pipeline ?target ~alias () =
  [ simplify; guards; simplify; hoist ~alias ]
  @ (match target with
    | Some target ->
        (* the splats the vectorizer made are loop invariant *)
        [
          vectorize ~alias ~target;
          block ~alias ~group:Ssa_opt_block.Auto;
          hoist ~alias;
        ]
    | None -> [ block ~alias ~group:Ssa_opt_block.Auto ])
  @ [ share ~alias; simplify ]

type report = (string * int) list

let verified ~after (p : Ssa_program.t) =
  match Err.payload (Ssa_verify.check p) with
  | Ok () -> p
  | Error e ->
      invalid_arg
        (Fmt.str "Ssa_opt: %s returned a program that does not verify: %a" after
           Ssa_verify.pp_error e)

let run ?(alias = Ssa_effects.Conservative) ?target ?passes (p : Ssa_program.t)
    =
  let passes =
    match passes with Some ps -> ps | None -> pipeline ?target ~alias ()
  in
  let counts = Hashtbl.create 8 in
  let rec rounds p n =
    let p, changed =
      List.fold_left
        (fun (p, changed) pass ->
          let p', c = pass.run p in
          if c then (
            Hashtbl.replace counts pass.name
              (1 + Option.value (Hashtbl.find_opt counts pass.name) ~default:0);
            (verified ~after:pass.name p', true))
          else (p, changed))
        (p, false) passes
    in
    if changed && n > 0 then rounds p (n - 1) else p
  in
  let p = rounds p 8 in
  let names =
    List.sort_uniq String.compare (List.map (fun pass -> pass.name) passes)
  in
  ( p,
    List.map
      (fun name ->
        (name, Option.value (Hashtbl.find_opt counts name) ~default:0))
      names )
