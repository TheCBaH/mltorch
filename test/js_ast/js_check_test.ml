open Js_ast

let id = Ident.v
let v s = Var (id s)

let program ?(prelude = []) ?(params = []) body =
  {
    Program.prelude;
    entry = { Func.name = id "k"; params = List.map id params; body };
  }

let report p =
  match Js_check.closed p with
  | Ok () -> Fmt.pr "closed@."
  | Error faults -> List.iter (Fmt.pr "%a@." Js_check.Fault.pp) faults

let%expect_test "a well-scoped program is closed" =
  report
    (program ~params:[ "a"; "n" ]
       ~prelude:
         [
           Stmt.Const
             (id "tab", New (Global Global.Float32_array, [ Number 4. ]));
           Stmt.Function
             {
               Func.name = id "helper";
               params = [ id "x" ];
               body = [ Stmt.Return (Some (Index (v "tab", v "x"))) ];
             };
         ]
       [
         Stmt.Let (id "s", Number 0.);
         Stmt.For
           {
             var = id "i";
             init = Number 0.;
             test = Binary (Lt, v "i", v "n");
             body =
               [
                 Stmt.Let (id "t", Call (v "helper", [ v "i" ]));
                 Stmt.Assign (Lvar (id "s"), Plus_eq, v "t");
               ];
           };
         Stmt.Return (Some (v "s"));
       ]);
  [%expect {| closed |}]

let%expect_test "an undeclared use is reported, and free names the same" =
  let body = [ Stmt.Assign (Lvar (id "y"), Eq, Binary (Add, v "x", v "z")) ] in
  report (program body);
  Fmt.pr "%a@."
    Fmt.(list ~sep:(any ", ") Ident.pp)
    (Ident.Set.elements (Js_check.free body));
  [%expect
    {|
    unbound identifier y
    unbound identifier x
    unbound identifier z
    x, y, z
    |}]

let%expect_test "a use before its let is unbound; a later function is hoisted" =
  report
    (program
       [
         Stmt.Expr (Call (v "later", []));
         Stmt.Expr (v "x");
         Stmt.Let (id "x", Number 1.);
         Stmt.Function { Func.name = id "later"; params = []; body = [] };
       ]);
  [%expect {| unbound identifier x |}]

let%expect_test "a duplicate let in one block, and a let over a parameter" =
  report
    (program ~params:[ "a" ]
       [
         Stmt.Let (id "x", Number 1.);
         Stmt.Let (id "x", Number 2.);
         Stmt.Let (id "a", Number 3.);
       ]);
  [%expect
    {|
    duplicate declaration of x
    duplicate declaration of a
    |}]

let%expect_test "the For variable is not visible after the loop" =
  report
    (program ~params:[ "n" ]
       [
         Stmt.For
           {
             var = id "i";
             init = Number 0.;
             test = Binary (Lt, v "i", v "n");
             body = [];
           };
         Stmt.Return (Some (v "i"));
       ]);
  [%expect {| unbound identifier i |}]

let%expect_test "the For variable is not visible in its own initialiser" =
  report
    (program
       [ Stmt.For { var = id "i"; init = v "i"; test = Bool false; body = [] } ]);
  [%expect {| unbound identifier i |}]

let%expect_test "shadowing in a nested block is accepted" =
  report
    (program ~params:[ "a" ]
       [
         Stmt.Let (id "x", Number 1.);
         Stmt.If
           ( v "a",
             [ Stmt.Let (id "x", Number 2.); Stmt.Expr (v "x") ],
             [ Stmt.Let (id "x", Number 3.) ] );
         Stmt.For
           {
             var = id "x";
             init = Number 0.;
             test = Bool false;
             body = [ Stmt.Let (id "x", Number 4.) ];
           };
       ]);
  [%expect {| closed |}]

let%expect_test "a let declared in a branch is not visible after it" =
  report
    (program ~params:[ "a" ]
       [
         Stmt.If (v "a", [ Stmt.Let (id "t", Number 1.) ], []);
         Stmt.Expr (v "t");
       ]);
  [%expect {| unbound identifier t |}]

let%expect_test "free of a helper names what its body reads" =
  let helper =
    Stmt.Function
      {
        Func.name = id "h";
        params = [ id "x" ];
        body =
          [
            Stmt.Return (Some (Binary (Add, v "x", Call (v "g", [ v "tab" ]))));
          ];
      }
  in
  Fmt.pr "%a@."
    Fmt.(list ~sep:(any ", ") Ident.pp)
    (Ident.Set.elements (Js_check.free [ helper ]));
  [%expect {| g, tab |}]
