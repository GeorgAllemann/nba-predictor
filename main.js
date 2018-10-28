import fs, { stat } from 'fs'
import Papa from 'papaparse'
import nba from 'nba'
import brain, { likely } from 'brain.js'

const east = []
const west = []

async function main () {

  /*
  try {
    const test = await nba.stats.teamSplits({ TeamID: 1610612739, PaceAdjust: 'Y', Rank: 'Y' })
    console.log(test)
  } catch (error) {
    console.error(error)
  }
  */

  const config = {
    inputSize: 20,
    inputRange: 20,
    hiddenSizes: [20, 20, 5],
    outputSize: 20,
    learningRate: 0.3,
    logPeriod: 1000,
    iterations: 20000,
    errorThresh: 0.1
  }

  const net = new brain.NeuralNetwork(config)

  const seasonList = ['2014-15', '2015-16', '2016-17']

  const trainingList = []
  seasonList.forEach((season) => {
    trainingList.push(trainSeason(season))
  })

  const result = await Promise.all(trainingList)
  const trainingData = [].concat.apply([], result)
  const trainResult = net.train(trainingData, {
    log: true,
  })
  console.log('trainResult', trainResult)

  const testData = await trainSeason('2017-18')
  console.log('testData', testData)
  let rightPrediction = 0
  let counter = 0
  testData.forEach((dataPoint) => {
    console.log('datapoint', dataPoint)
    const prediction = net.run(dataPoint.input)
    console.log('prediction', prediction)
    if (prediction > 0.5 || prediction < 0.5) {
      counter += 1
      if (prediction > 0.5 && dataPoint.output[0] === 1) rightPrediction += 1
      if (prediction < 0.5 && dataPoint.output[0] === 0) rightPrediction += 1
    }
  })
  console.log('Games in total: ', testData.length)
  console.log('counter', counter)
  console.log('percentage output ', rightPrediction / counter)

  fs.writeFileSync('./data.json', JSON.stringify(testData) , 'utf-8'); 

  const run = net.toFunction();
  console.log(run.toString());
}

function trainSeason (season) {
  return new Promise((resolve, reject) => {
    fs.readFile(`./data/season-${season}.csv`, 'utf8', (err, data) => {
      if (err) console.error(err)
      const games = Papa.parse(data, {
        header: true
      }).data

      const gamePairs = {}

      games.forEach((game) => {
        if (!game.GAME_ID) return
        if (!gamePairs[game.GAME_ID]) gamePairs[game.GAME_ID] = []
        gamePairs[game.GAME_ID].push(game)
      })

      const trainingData = []
      Object.values(gamePairs).forEach((gamePair) => {
        const [ homeTeam, awayTeam ] = gamePair
        const lastHomeGames = getLast10Games(games, homeTeam.GAME_DATE, homeTeam.TEAM_ID)
        const lastAwayGames = getLast10Games(games, awayTeam.GAME_DATE, awayTeam.TEAM_ID)
        if (lastHomeGames.length > 9 && lastAwayGames.length > 9) {
          const homeStats = getAvarageStatistics(lastHomeGames)
          const awayStats = getAvarageStatistics(lastAwayGames)

          const input = []
          Object.entries(homeStats).map(([key, value]) => {
            input.push(value - awayStats[key])
          })

          const output = [homeTeam.WL === 'W' ? 1 : 0]
          trainingData.push({
            input,
            output,
            date: homeTeam.GAME_DATE,
            home: homeTeam.TEAM_NAME,
            away: awayTeam.TEAM_NAME,
          })
        }
      })
      resolve(trainingData)
    })
  })
}

function getAvarageStatistics(last10Games) {
  const stats = {
    wins: 0,
    plusMinus: 0,
    fieldGoalPercentage: 0,
    threePointerPercentage: 0,
    // turnoversToAssists: 0,
    offensiveReboundPercentage: 0
    // freethrowPercentage: 0
  }

  last10Games.forEach((game) => {
    if (game.WL === 'W') stats.wins += 1
    stats.plusMinus += parseFloat(game.PLUS_MINUS) / 10
    stats.fieldGoalPercentage += parseFloat(game.FG_PCT) / 10
    stats.threePointerPercentage += parseFloat(game.FG3_PCT) / 10
    // stats.turnoversToAssists += (parseFloat(game.TOV) / parseFloat(game.AST)) / 10
    stats.offensiveReboundPercentage += (parseFloat(game.OREB) / parseFloat(game.REB)) / 10
    // stats.freethrowPercentage += parseFloat(game.FT_PCT) / 10
  })
  return stats
}

function getLast10Games (gameList, date, teamId) {
  return gameList.filter(game => game.TEAM_ID === teamId && game.GAME_DATE < date).slice(-10)
}

console.log('everything will start now')
main()
