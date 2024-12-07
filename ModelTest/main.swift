//
//  main.swift
//  ModelTest
//
//  Created by guinmoon on 20.05.2023.
//

import Foundation
import llmfarm_core
import llmfarm_core_cpp

let maxOutputLength:Int32 = 2000 //250
var total_output = 0
var session_tokens: [Int32] = []
var ai: AI? = nil

func mainCallback(_ str: String, _ time: Double) -> Bool {
    print("\(str)",terminator: "")
    total_output += str.count
    // print(total_output)
    // if  total_output>maxOutputLength {        
    //     ai!.flagExit = true        
    //     return true
    // }
    return false
}

func set_promt_format(ai: inout AI?) throws -> Bool{
    ai!.model?.contextParams.promptFormat = .None
    return true
}

func main(){
    print("Hello.")
    var input_text = "State the meaning of life."
    var modelInference:ModelInference
    ai = AI(_modelPath: "",_chatName: "chat")
    
    ai!.modelPath = "/Users/marcushobbs/Library/Containers/com.marcussatellite.LLMFarm/Data/Documents/models/Llama-3.2-3B-Instruct-Q4_K_S.gguf"
    //ai.modelPath = "/Users/marcushobbs/Library/Containers/com.marcussatellite.LLMFarm/Data/Documents/models/Llama-3.2-3B-Instruct-unsloth-Q4_K_M-IZ-04.gguf"
        modelInference = ModelInference.LLama_gguf
    //
    var params:ModelAndContextParams = .default
    params.context = 2048
    params.n_threads = 4
    //
    params.use_metal = true
    params.n_predict = maxOutputLength
    params.flash_attn = false
    // params.add_bos_token = false
    // params.add_eos_token = true
    params.parse_special_tokens = true
    
    input_text = "Write story about Artem."
    do{
        ai!.initModel(modelInference,contextParams: params)
        if ai!.model == nil{
            print( "Model load eror.")
            exit(2)
        }
        try ai!.loadModel_sync()
        
        
        var output: String?
        try ExceptionCather.catchException {
            output = try? ai!.model?.predict(input_text, mainCallback) ?? ""
        }

        print(output ?? "")
    } catch {
        print (error)
        return
    }
}

main()
