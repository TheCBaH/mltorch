(* See arena_run.mli. *)

open Graph_ir

type error =
  [ Arena.error
  | Arena_plan.error
  | Eval_direct.error
  | `Peak_bytes_overflow of Tensor_id.t ]

let pp_error ppf : [< error ] -> unit = function
  | #Arena.error as e -> Arena.pp_error ppf e
  | #Eval_direct.error as e -> Eval_direct.pp_error ppf e
  | `Arena_over_limit { Arena_plan.Over_limit.kind; numel; bytes; limit } ->
      Format.fprintf ppf
        "arena: the %a pool needs %Ld cells (%Ld bytes), over %s"
        Alloc_script.Kind.pp kind numel bytes
        (match limit with
        | Arena_plan.Over_limit.Bytes b -> Printf.sprintf "%Ld bytes" b
        | Arena_plan.Over_limit.Elements n -> Printf.sprintf "%Ld cells" n)
  | `Arena_placement _ -> Format.pp_print_string ppf "arena: placement failed"
  | `Arena_script id ->
      Format.fprintf ppf "arena: inconsistent script at t%d"
        (Tensor_id.to_int id)
  | `Peak_bytes_overflow id ->
      Format.fprintf ppf "arena: byte total overflows at t%d"
        (Tensor_id.to_int id)

type outcome = Arena of Arena.t | Release_only of error

let plan_and_create ?limits ?budget ?poison ?retain ~admission g =
  let open Err.Syntax in
  let* script =
    Eval_direct.dry_run ?retain g |> Err.map_error (fun e -> (e :> error))
  in
  let* plan =
    Arena_plan.create ?limits ?budget script
    |> Err.map_error (fun e -> (e :> error))
  in
  let* () =
    match admission with
    | Arena.Admission.Best_effort -> Err.return ()
    | Arena.Admission.Required budget ->
        let* footprint =
          Arena.footprint plan |> Err.map_error (fun e -> (e :> error))
        in
        if Int64.compare footprint budget > 0 then
          Err.fail ~pos:__POS__
            (`Over_budget { Arena.Over_budget.footprint; budget })
        else Err.return ()
  in
  Arena.create ?poison plan |> Err.map_error (fun e -> (e :> error))

(* The row is data for the caller's report, not an error of this run: under
   [Best_effort] a failure here means the run goes ahead without an arena. *)
let decline (e : error Err.Error.t) =
  match Err.export ~pos:__POS__ (Error e) with
  | Error row -> Release_only row
  | Ok _ -> assert false

let acquire ?limits ?budget ?poison ?retain ~admission g =
  match
    (plan_and_create ?limits ?budget ?poison ?retain ~admission g, admission)
  with
  | Ok arena, _ -> Err.return (Arena arena)
  | Error e, Arena.Admission.Best_effort -> Err.return (decline e)
  | (Error _ as e), Arena.Admission.Required _ -> (e :> (outcome, error) Err.t)

module Report = struct
  type t = {
    pool_bytes : int64;
    out_of_arena_bytes : int64;
    copies : Arena.Copies.t;
  }
end

let report arena =
  let open Err.Syntax in
  let plan = Arena.plan arena in
  let+ out_of_arena_bytes =
    Alloc_script.out_of_arena_bytes (Arena_plan.script plan)
  in
  {
    Report.pool_bytes = (Arena_plan.stats plan).Arena_plan.Stats.pool_bytes;
    out_of_arena_bytes;
    copies = Arena.copies arena;
  }

module Outcome = struct
  type t = Declined of error | Used of Report.t
end

let with_arena ?limits ?budget ?poison ?retain ~admission g
    (f : Arena.t option -> ('a, error) Err.t) : ('a * Outcome.t, error) Err.t =
  let open Err.Syntax in
  let* outcome = acquire ?limits ?budget ?poison ?retain ~admission g in
  match outcome with
  | Release_only reason ->
      let+ r = f None in
      (r, Outcome.Declined reason)
  | Arena arena ->
      let* r = f (Some arena) in
      let+ used =
        Err.map_error (fun (`Peak_bytes_overflow _ as e) -> e) (report arena)
      in
      (r, Outcome.Used used)
