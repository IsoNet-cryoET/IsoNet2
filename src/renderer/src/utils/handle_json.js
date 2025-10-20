function safeClone(obj) {
    return typeof structuredClone === 'function'
      ? structuredClone(obj)
      : JSON.parse(JSON.stringify(obj));
  }

function toCommand(data_raw, id) {
    console.log("data here1")
    console.log(data_raw)
    const data = safeClone(data_raw);
    if (!data || typeof data !== 'object') throw new Error('toCommand: invalid data');
    
    if (data.type === 'prepare_star' && data.hasOwnProperty('even_odd_input') && data.even_odd_input)
        data.full = 'None'
    if (data.type === 'prepare_star' && data.hasOwnProperty('even_odd_input') && ! data.even_odd_input){
        data.odd = 'None'
        data.evn = 'None'
    }
    if (data.type !== 'prepare_star' && data.type !== 'star2json'){
        data.output_dir = data.type+"/job"+id+"_"+data.name
    }
    console.log(id)
    console.log(data.output_dir)

    let result = ''
    const pyBoolMap = { true: 'True', false: 'False', null: 'None' };
    for (const key in data) {
        let value = data[key]
        if (key === 'status' || key === 'even_odd_input' || key == "id" || key === "name") continue
        if (key === 'type' ) {
            result = `${value}${result}`
        } else {
            value = pyBoolMap[value] ?? value;
            result += ` --${key} ${value}`
        }        
    }
    return "isonet.py "+result
}

export default toCommand