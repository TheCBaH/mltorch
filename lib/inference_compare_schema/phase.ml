module W = Wire

type t =
  | Commit
  | Execution
  | Export
  | Import
  | Input_check
  | Planning
  | Preparation

let decode, encode =
  W.enum ~kind:"phase"
    [
      (Commit, "commit");
      (Execution, "execution");
      (Export, "export");
      (Import, "import");
      (Input_check, "input_check");
      (Planning, "planning");
      (Preparation, "preparation");
    ]
