(* Symbolic integers for PyTorch-style dynamic shapes: an axis size may be a
   variable (s0, s1, …) or an affine expression over them, carried with a
   ShapeEnv of per-variable constraints (range + divisibility). Direct execution
   binds the variables to concrete ints (checking the guards) before allocating;
   the symbolic/footprint layers keep them symbolic. See native_tensor_design §4. *)

type t = Add of t * t | Const of int | Scale of int * t | Var of string

let const n = Const n
let var name = Var name
let add a b = Add (a, b)
let scale k a = Scale (k, a)

let rec pp fmt = function
  | Add (a, b) -> Fmt.pf fmt "%a + %a" pp a pp b
  | Const n -> Fmt.int fmt n
  | Scale (k, a) -> Fmt.pf fmt "%d*%a" k pp a
  | Var v -> Fmt.string fmt v

(* A variable's declared constraints. [divisor = 1] means "no divisibility
   requirement"; [hi] is exclusive. *)
type var_info = { lo : int; hi : int; divisor : int }
type env = { decls : (string * var_info) list; binds : (string * int) list }

let empty = { decls = []; binds = [] }

let declare env name ~lo ~hi ?(divisor = 1) () =
  { env with decls = (name, { lo; hi; divisor }) :: env.decls }

(* Error set owned by this module (the eval/binding phase): a guard violation
   when binding a dynamic-shape variable. *)
type range = { name : string; value : int; lo : int; hi : int }
type div = { name : string; value : int; divisor : int }
type error = [ `Not_divisible of div | `Out_of_range of range ]

let pp_error ppf : error -> unit = function
  | `Not_divisible { name; value; divisor } ->
      Fmt.pf ppf "symint %s = %d not divisible by %d" name value divisor
  | `Out_of_range { name; value; lo; hi } ->
      Fmt.pf ppf "symint %s = %d out of [%d, %d)" name value lo hi

(* Bind a variable to a concrete value, checking it against the declared
   constraints (mirrors dynamo's guard check). [Error] on a guard violation. *)
let bind env name value =
  let open Err.Syntax in
  let add () = Err.return { env with binds = (name, value) :: env.binds } in
  match List.assoc_opt name env.decls with
  | None -> add ()
  | Some { lo; hi; divisor } ->
      let* () =
        if value < lo || value >= hi then
          Err.fail (`Out_of_range { name; value; lo; hi })
        else Err.return ()
      in
      let* () =
        if value mod divisor <> 0 then
          Err.fail (`Not_divisible { name; value; divisor })
        else Err.return ()
      in
      add ()

(* [Some n] once every variable in the expression is bound. *)
let rec eval env = function
  | Add (a, b) -> (
      match (eval env a, eval env b) with
      | Some x, Some y -> Some (x + y)
      | _ -> None)
  | Const n -> Some n
  | Scale (k, a) -> (
      match eval env a with Some x -> Some (k * x) | None -> None)
  | Var v -> List.assoc_opt v env.binds

(* An axis size: concrete, or a symbolic expression awaiting binding. *)
type size = Static of int | Sym of t

let pp_size fmt = function
  | Static n -> Format.fprintf fmt "%d" n
  | Sym s -> pp fmt s

let resolve env = function Static n -> Some n | Sym s -> eval env s
