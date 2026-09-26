open Js_ast
module Set = Ident.Set

module Fault = struct
  type t = Duplicate of Ident.t | Unbound of Ident.t

  let pp ppf = function
    | Duplicate id -> Fmt.pf ppf "duplicate declaration of %a" Ident.pp id
    | Unbound id -> Fmt.pf ppf "unbound identifier %a" Ident.pp id
end

let rec expr report scope e =
  let go = expr report scope in
  match e with
  | Var id -> if not (Set.mem id scope) then report (Fault.Unbound id)
  | Array es -> List.iter go es
  | Binary (_, a, b) | Index (a, b) ->
      go a;
      go b
  | Call (f, args) | New (f, args) ->
      go f;
      List.iter go args
  | Cond (c, a, b) ->
      go c;
      go a;
      go b
  | Member (a, _) | Unary (_, a) -> go a
  | Object fields -> List.iter (fun (_, v) -> go v) fields
  | Bigint _ | Bool _ | Global _ | Null | Number _ | String _ -> ()

let lvalue report scope = function
  | Lvar id -> expr report scope (Var id)
  | Lindex (a, i) ->
      expr report scope a;
      expr report scope i

let declared : Stmt.t -> Ident.t option = function
  | Stmt.Const (id, _) | Stmt.Let (id, _) -> Some id
  | Stmt.Function f -> Some f.Func.name
  | Stmt.Assign _ | Stmt.Expr _ | Stmt.For _ | Stmt.If _ | Stmt.Return _ -> None

(* A block: its own declarations (and [bound], its parameters) form one
   namespace, so a repeat is a fault. A function body sees the block's every
   name, since it runs after the block is set up; a straight-line statement
   sees the functions and only the [let]/[const] before it. *)
let rec block report scope ~bound stmts =
  let names = List.filter_map declared stmts in
  ignore
    (List.fold_left
       (fun seen id ->
         if Set.mem id seen then report (Fault.Duplicate id);
         Set.add id seen)
       Set.empty (bound @ names));
  let functions =
    List.filter_map
      (function Stmt.Function f -> Some f.Func.name | _ -> None)
      stmts
  in
  let with_ ids s = List.fold_left (fun s id -> Set.add id s) s ids in
  let outer = with_ bound scope in
  let everything = with_ names outer in
  ignore
    (List.fold_left
       (fun cur (s : Stmt.t) ->
         match s with
         | Stmt.Assign (lv, _, e) ->
             lvalue report cur lv;
             expr report cur e;
             cur
         | Stmt.Const (id, e) | Stmt.Let (id, e) ->
             expr report cur e;
             Set.add id cur
         | Stmt.Expr e ->
             expr report cur e;
             cur
         | Stmt.For { var; init; test; body } ->
             expr report cur init;
             let inner = Set.add var cur in
             expr report inner test;
             block report inner ~bound:[] body;
             cur
         | Stmt.Function f ->
             block report everything ~bound:f.Func.params f.Func.body;
             cur
         | Stmt.If (c, yes, no) ->
             expr report cur c;
             block report cur ~bound:[] yes;
             block report cur ~bound:[] no;
             cur
         | Stmt.Return e ->
             Option.iter (expr report cur) e;
             cur)
       (with_ functions outer) stmts)

let collect ~bound stmts =
  let faults = ref [] in
  block (fun f -> faults := f :: !faults) Set.empty ~bound stmts;
  List.rev !faults

let closed (p : Program.t) =
  match
    collect ~bound:[] (p.Program.prelude @ [ Stmt.Function p.Program.entry ])
  with
  | [] -> Ok ()
  | faults -> Error faults

let free stmts =
  List.fold_left
    (fun acc -> function
      | Fault.Unbound id -> Set.add id acc | Fault.Duplicate _ -> acc)
    Set.empty (collect ~bound:[] stmts)
